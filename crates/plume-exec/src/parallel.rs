//! Parallel parquet scan — Week 13.
//!
//! `ParallelParquetScanExec` reads row groups in parallel using rayon. The
//! sequential `ParquetScanExec` reads row groups one after another; on a
//! file with N row groups and a multi-core machine, that leaves cores
//! idle. This operator dispatches each surviving row group to a rayon
//! task and streams batches as they arrive.
//!
//! Trade-offs vs the sequential scan:
//!
//! - **Output order is undefined.** Row group 7 may finish before row
//!   group 0. Operators downstream that don't care about input order
//!   (Filter, HashAggregateExec, AggregateExec, ProjectionExec) work
//!   fine. `LimitExec` *does* care — applying a limit on the parallel
//!   stream cuts off whichever row groups happen to land first, which
//!   may not be the file's first N rows. Don't compose Limit on top of
//!   this scan unless that's what you want.
//!
//! - **Each task opens its own File handle.** The parquet builder takes
//!   ownership of a `File`, and the per-worker code path needs its own
//!   metadata + ProjectionMask. The cost is N open/close + N metadata
//!   reads instead of one; for files with many row groups that's
//!   negligible vs the data-reading work, but for tiny files it's a wash.
//!
//! - **Pushdown predicates and projection are honoured per worker.**
//!   Each worker reads only its assigned row groups (using
//!   `with_row_groups(vec![idx])`) so we don't double-prune.
//!
//! Opt-in: the planner does not auto-pick this executor. Callers who
//! want it construct it directly.

use std::fs::File;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::statistics::Statistics;
use plume_core::{BinaryOperator, Expr, ScalarValue};
use rayon::prelude::*;

use crate::eval::{EvalError, EvalResult};
use crate::exec::{BatchStream, ExecutionPlan};

const DEFAULT_BATCH_SIZE: usize = 8192;

pub struct ParallelParquetScanExec {
    path: PathBuf,
    schema: SchemaRef,
    projection: Option<Vec<String>>,
    pushdown_predicates: Vec<Expr>,
    batch_size: usize,
    /// Cap on parallel workers. None = use rayon's global pool default
    /// (typically `num_cpus`). Set to 1 for a single-threaded sanity run.
    max_workers: Option<usize>,
}

impl ParallelParquetScanExec {
    pub fn new(path: PathBuf, schema: SchemaRef) -> Self {
        Self {
            path,
            schema,
            projection: None,
            pushdown_predicates: Vec::new(),
            batch_size: DEFAULT_BATCH_SIZE,
            max_workers: None,
        }
    }

    pub fn with_projection(
        path: PathBuf,
        projected_schema: SchemaRef,
        projected_columns: Vec<String>,
    ) -> Self {
        Self {
            path,
            schema: projected_schema,
            projection: Some(projected_columns),
            pushdown_predicates: Vec::new(),
            batch_size: DEFAULT_BATCH_SIZE,
            max_workers: None,
        }
    }

    pub fn with_pushdown_predicates(mut self, preds: Vec<Expr>) -> Self {
        self.pushdown_predicates = preds;
        self
    }

    pub fn with_max_workers(mut self, n: usize) -> Self {
        self.max_workers = Some(n);
        self
    }
}

impl ExecutionPlan for ParallelParquetScanExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        // ---- one cheap metadata pass on the main thread to compute
        //      surviving row groups under any pushdown predicates ----------
        let surviving = surviving_row_groups(&self.path, &self.pushdown_predicates)?;

        // Snapshot the per-worker config into owned values so the spawned
        // thread doesn't need to borrow self.
        let path = self.path.clone();
        let projection = self.projection.clone();
        let batch_size = self.batch_size;
        let max_workers = self.max_workers;

        // Channel-based fan-in: each rayon task pushes its batches into the
        // same mpsc, and the foreground iterator pulls them out. The thread
        // that drives rayon hands ownership of the receiver back as the
        // BatchStream.
        let (tx, rx) = mpsc::channel::<EvalResult<RecordBatch>>();

        thread::Builder::new()
            .name("plume-par-scan-driver".into())
            .spawn(move || {
                let pool = match max_workers {
                    Some(n) => rayon::ThreadPoolBuilder::new().num_threads(n).build().ok(),
                    None => None,
                };

                let work = || {
                    surviving.par_iter().for_each(|&rg_idx| {
                        let res = read_row_group(&path, &projection, batch_size, rg_idx, &tx);
                        if let Err(e) = res {
                            // Ignore send errors — the receiver may have
                            // dropped after the consumer hit a limit / err.
                            let _ = tx.send(Err(e));
                        }
                    });
                };

                if let Some(pool) = pool {
                    pool.install(work);
                } else {
                    work();
                }
                // tx is dropped here, ending the receiver's iteration.
            })
            .expect("spawn parallel scan driver");

        Ok(Box::new(rx.into_iter()))
    }
}

/// Read a single row group's batches and push them into `tx`.
fn read_row_group(
    path: &std::path::Path,
    projection: &Option<Vec<String>>,
    batch_size: usize,
    rg_idx: usize,
    tx: &mpsc::Sender<EvalResult<RecordBatch>>,
) -> EvalResult<()> {
    let file = File::open(path)?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(batch_size)
        .with_row_groups(vec![rg_idx]);

    if let Some(cols) = projection {
        let parquet_schema = builder.parquet_schema();
        let root = parquet_schema.root_schema();
        let mut indices = Vec::with_capacity(cols.len());
        for name in cols {
            let idx = root
                .get_fields()
                .iter()
                .position(|f| f.name() == name)
                .ok_or_else(|| EvalError::MissingColumn(name.clone()))?;
            indices.push(idx);
        }
        let mask = parquet::arrow::ProjectionMask::roots(parquet_schema, indices);
        builder = builder.with_projection(mask);
    }

    let reader = builder.build()?;
    for batch in reader {
        let result = batch.map_err(EvalError::from);
        if tx.send(result).is_err() {
            // Consumer disappeared (e.g. surrounding query short-circuited);
            // stop reading.
            break;
        }
    }
    Ok(())
}

/// Compute the row groups that survive `predicates` against the file's
/// metadata. Identical logic to ParquetScanExec's row_group_can_satisfy
/// but lifted out so we can reuse it; if neither predicates nor stats are
/// in play, returns 0..N.
fn surviving_row_groups(
    path: &std::path::Path,
    predicates: &[Expr],
) -> EvalResult<Arc<Vec<usize>>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let metadata = builder.metadata();
    let total = metadata.num_row_groups();
    if predicates.is_empty() {
        return Ok(Arc::new((0..total).collect()));
    }
    let parquet_schema = builder.parquet_schema();
    let root = parquet_schema.root_schema();
    let kept: Vec<usize> = (0..total)
        .filter(|&rg_idx| {
            let rg = metadata.row_group(rg_idx);
            predicates
                .iter()
                .all(|pred| row_group_can_satisfy(pred, rg, root))
        })
        .collect();
    Ok(Arc::new(kept))
}

// Mirror of operators.rs::row_group_can_satisfy. Lifted here rather than
// pub(crate)'d to keep operators.rs internals private; the duplication is
// minor (a few dozen lines) and the rule is exactly identical.
fn row_group_can_satisfy(
    pred: &Expr,
    rg: &parquet::file::metadata::RowGroupMetaData,
    parquet_root: &parquet::schema::types::Type,
) -> bool {
    let Expr::BinaryOp { left, op, right } = pred else {
        return true;
    };
    let Expr::Column(name) = left.as_ref() else {
        return true;
    };
    let Expr::Literal(lit) = right.as_ref() else {
        return true;
    };
    let Some(col_idx) = parquet_root
        .get_fields()
        .iter()
        .position(|f| f.name() == name)
    else {
        return true;
    };
    let chunk = rg.column(col_idx);
    let Some(stats) = chunk.statistics() else {
        return true;
    };
    if let Some(null_count) = stats.null_count_opt() {
        if null_count == rg.num_rows() as u64 {
            return false;
        }
    }
    match (stats, lit) {
        (Statistics::Int32(s), ScalarValue::Int32(v)) => check_ord(
            s.min_opt().copied().map(i64::from),
            s.max_opt().copied().map(i64::from),
            *op,
            *v as i64,
        ),
        (Statistics::Int32(s), ScalarValue::Int64(v)) => check_ord(
            s.min_opt().copied().map(i64::from),
            s.max_opt().copied().map(i64::from),
            *op,
            *v,
        ),
        (Statistics::Int64(s), ScalarValue::Int32(v)) => {
            check_ord(s.min_opt().copied(), s.max_opt().copied(), *op, *v as i64)
        }
        (Statistics::Int64(s), ScalarValue::Int64(v)) => {
            check_ord(s.min_opt().copied(), s.max_opt().copied(), *op, *v)
        }
        (Statistics::Float(s), ScalarValue::Float64(v)) => check_ord_f64(
            s.min_opt().copied().map(f64::from),
            s.max_opt().copied().map(f64::from),
            *op,
            *v,
        ),
        (Statistics::Double(s), ScalarValue::Float64(v)) => {
            check_ord_f64(s.min_opt().copied(), s.max_opt().copied(), *op, *v)
        }
        (Statistics::ByteArray(s), ScalarValue::Utf8(v)) => {
            let mn = s.min_opt().and_then(|b| std::str::from_utf8(b.data()).ok());
            let mx = s.max_opt().and_then(|b| std::str::from_utf8(b.data()).ok());
            check_ord_str(mn, mx, *op, v.as_str())
        }
        _ => true,
    }
}

fn check_ord<T: Ord + Copy>(min: Option<T>, max: Option<T>, op: BinaryOperator, lit: T) -> bool {
    use BinaryOperator::*;
    let (Some(mn), Some(mx)) = (min, max) else {
        return true;
    };
    match op {
        Eq => !(lit < mn || lit > mx),
        NotEq => true,
        Lt => mn < lit,
        LtEq => mn <= lit,
        Gt => mx > lit,
        GtEq => mx >= lit,
        _ => true,
    }
}

fn check_ord_f64(min: Option<f64>, max: Option<f64>, op: BinaryOperator, lit: f64) -> bool {
    use BinaryOperator::*;
    let (Some(mn), Some(mx)) = (min, max) else {
        return true;
    };
    if mn.is_nan() || mx.is_nan() || lit.is_nan() {
        return true;
    }
    match op {
        Eq => !(lit < mn || lit > mx),
        NotEq => true,
        Lt => mn < lit,
        LtEq => mn <= lit,
        Gt => mx > lit,
        GtEq => mx >= lit,
        _ => true,
    }
}

fn check_ord_str(min: Option<&str>, max: Option<&str>, op: BinaryOperator, lit: &str) -> bool {
    use BinaryOperator::*;
    let (Some(mn), Some(mx)) = (min, max) else {
        return true;
    };
    match op {
        Eq => !(lit < mn || lit > mx),
        NotEq => true,
        Lt => mn < lit,
        LtEq => mn <= lit,
        Gt => mx > lit,
        GtEq => mx >= lit,
        _ => true,
    }
}

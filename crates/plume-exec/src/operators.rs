//! Physical operators: CsvScan, Memory (test fixture), Filter, Projection, Limit.
//!
//! Each is a small struct that owns its inputs and produces a fresh batch
//! stream on every `execute()` call. None of them cache results.

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::compute::filter_record_batch;
use arrow::csv;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use plume_core::Expr;

use crate::compiled::{compile, CompiledExpr};
use crate::eval::{EvalError, EvalResult};
use crate::exec::{BatchStream, ExecutionPlan};
use crate::metrics::ScanMetrics;
use crate::pruning::surviving_row_groups;

const DEFAULT_BATCH_SIZE: usize = 8192;

// ---- CSV scan ----------------------------------------------------------

pub struct CsvScanExec {
    path: PathBuf,
    /// Output schema. When `projection` is `Some`, this is the file schema
    /// restricted to (and reordered by) those columns.
    schema: SchemaRef,
    /// Indices into the file's full schema, when set by the optimizer.
    /// `None` = read every column.
    projection: Option<Vec<usize>>,
    /// File's full schema. Required so arrow's CSV reader can parse every
    /// row in source order before the projection narrows it.
    file_schema: SchemaRef,
    batch_size: usize,
    has_header: bool,
}

impl CsvScanExec {
    pub fn new(path: PathBuf, schema: SchemaRef) -> Self {
        Self {
            path,
            schema: schema.clone(),
            projection: None,
            file_schema: schema,
            batch_size: DEFAULT_BATCH_SIZE,
            has_header: true,
        }
    }

    /// Construct with a projection. `file_schema` is the unprojected schema
    /// of the source file; `projected_columns` are the names to keep.
    pub fn with_projection(
        path: PathBuf,
        file_schema: SchemaRef,
        projected_columns: &[String],
    ) -> EvalResult<Self> {
        let mut indices = Vec::with_capacity(projected_columns.len());
        let mut fields = Vec::with_capacity(projected_columns.len());
        for name in projected_columns {
            let idx = file_schema
                .fields()
                .iter()
                .position(|f| f.name() == name)
                .ok_or_else(|| EvalError::MissingColumn(name.clone()))?;
            indices.push(idx);
            fields.push((*file_schema.field(idx)).clone());
        }
        let projected_schema = Arc::new(arrow_schema::Schema::new(fields));
        Ok(Self {
            path,
            schema: projected_schema,
            projection: Some(indices),
            file_schema,
            batch_size: DEFAULT_BATCH_SIZE,
            has_header: true,
        })
    }
}

impl ExecutionPlan for CsvScanExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        let file = File::open(&self.path)?;
        let mut builder = csv::ReaderBuilder::new(self.file_schema.clone())
            .with_header(self.has_header)
            .with_batch_size(self.batch_size);
        if let Some(indices) = &self.projection {
            builder = builder.with_projection(indices.clone());
        }
        let reader = builder.build(file)?;
        let mapped = reader.map(|r| r.map_err(EvalError::from));
        Ok(Box::new(mapped))
    }
}

// ---- Parquet scan ------------------------------------------------------

pub struct ParquetScanExec {
    path: PathBuf,
    schema: SchemaRef,
    /// Column names to project, in output order. None = read every column.
    /// We intentionally store names rather than precomputed indices: the
    /// parquet schema and ProjectionMask live behind the file metadata,
    /// which we re-read on every execute() call.
    projection: Option<Vec<String>>,
    /// Conjunction of `col OP literal` predicates from the predicate
    /// pushdown optimizer. Used as a row-group-pruning *hint*: if a row
    /// group's min/max stats prove a predicate cannot be satisfied for
    /// any value in that group, we skip reading the group entirely.
    /// The Filter that originally held them stays in the plan and still
    /// does the per-row check.
    pushdown_predicates: Vec<Expr>,
    /// Optional row-count cap from the limit-pushdown optimizer. When set,
    /// the scan tells the parquet reader to stop after `limit` rows.
    limit: Option<usize>,
    batch_size: usize,
    /// Counters for the most recent execute(); see `crate::metrics`.
    metrics: Arc<ScanMetrics>,
}

impl ParquetScanExec {
    pub fn new(path: PathBuf, schema: SchemaRef) -> Self {
        Self {
            path,
            schema,
            projection: None,
            pushdown_predicates: Vec::new(),
            limit: None,
            batch_size: DEFAULT_BATCH_SIZE,
            metrics: Arc::default(),
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
            limit: None,
            batch_size: DEFAULT_BATCH_SIZE,
            metrics: Arc::default(),
        }
    }

    pub fn with_pushdown_predicates(mut self, preds: Vec<Expr>) -> Self {
        self.pushdown_predicates = preds;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Shared handle to this scan's metrics. Keep a clone before wrapping
    /// the scan in `Arc<dyn ExecutionPlan>`; read it after the stream is
    /// drained.
    pub fn metrics(&self) -> Arc<ScanMetrics> {
        Arc::clone(&self.metrics)
    }
}

impl ExecutionPlan for ParquetScanExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        let file = File::open(&self.path)?;
        let mut builder =
            ParquetRecordBatchReaderBuilder::try_new(file)?.with_batch_size(self.batch_size);
        if let Some(n) = self.limit {
            builder = builder.with_limit(n);
        }

        // ---- row-group pruning via pushdown predicates ------------------
        // With no predicates every row group survives.
        let total = builder.metadata().num_row_groups();
        let surviving = surviving_row_groups(
            builder.metadata(),
            builder.parquet_schema().root_schema(),
            &self.pushdown_predicates,
        );
        self.metrics.reset_for_execute(
            builder.metadata(),
            builder.parquet_schema(),
            self.projection.as_deref(),
            &surviving,
        );

        // Only set with_row_groups if we actually pruned some — passing
        // every group is equivalent to not setting it but lets the
        // reader take its default code path.
        if surviving.len() < total {
            builder = builder.with_row_groups(surviving);
        }

        // ---- column projection (Week 8) ---------------------------------
        let reader = match &self.projection {
            None => builder.build()?,
            Some(cols) => {
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
                builder.with_projection(mask).build()?
            }
        };
        let metrics = Arc::clone(&self.metrics);
        Ok(Box::new(reader.map(move |r| {
            let batch = r.map_err(EvalError::from)?;
            metrics.add_rows_decoded(batch.num_rows());
            Ok(batch)
        })))
    }
}

// ---- in-memory scan (test scaffolding) ---------------------------------

pub struct MemoryExec {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
}

impl MemoryExec {
    pub fn new(batches: Vec<RecordBatch>, schema: SchemaRef) -> Self {
        Self { batches, schema }
    }
}

impl ExecutionPlan for MemoryExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        // Hand back owned clones (RecordBatch is cheap to clone — Arc'd arrays).
        let owned = self.batches.clone();
        Ok(Box::new(owned.into_iter().map(Ok)))
    }
}

// ---- filter ------------------------------------------------------------

pub struct FilterExec {
    input: Arc<dyn ExecutionPlan>,
    /// Pre-compiled predicate (W11). Column indices already resolved against
    /// the input schema, so the per-batch hot path skips name lookups.
    compiled: CompiledExpr,
    schema: SchemaRef,
}

impl FilterExec {
    /// Construct from a logical `Expr`. The predicate is compiled once here
    /// against `input.schema()` and reused for every batch.
    pub fn new(input: Arc<dyn ExecutionPlan>, predicate: Expr) -> Self {
        let schema = input.schema();
        // Schema-build-time errors should already have been caught by the
        // logical plan layer (LogicalPlan::filter validates the predicate
        // resolves cleanly against the input schema). If somehow a bad
        // predicate slipped through, surface it loudly at construction.
        let compiled = compile(&predicate, schema.as_ref()).expect(
            "predicate failed to compile against input schema; \
                    LogicalPlan::filter should have caught this at plan time",
        );
        Self {
            input,
            compiled,
            schema,
        }
    }
}

impl ExecutionPlan for FilterExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        let stream = self.input.execute()?;
        let compiled = self.compiled.clone();
        Ok(Box::new(stream.map(move |batch| {
            let batch = batch?;
            let mask = compiled.evaluate_predicate(&batch)?;
            let filtered = filter_record_batch(&batch, &mask)?;
            Ok(filtered)
        })))
    }
}

// ---- projection --------------------------------------------------------

pub struct ProjectionExec {
    input: Arc<dyn ExecutionPlan>,
    compiled: Vec<CompiledExpr>,
    schema: SchemaRef,
}

impl ProjectionExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, exprs: Vec<Expr>, schema: SchemaRef) -> Self {
        let in_schema = input.schema();
        let compiled = exprs
            .iter()
            .map(|e| compile(e, in_schema.as_ref()))
            .collect::<EvalResult<Vec<_>>>()
            .expect(
                "projection expressions failed to compile against input schema; \
                    LogicalPlan::project should have caught this at plan time",
            );
        Self {
            input,
            compiled,
            schema,
        }
    }
}

impl ExecutionPlan for ProjectionExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        let stream = self.input.execute()?;
        let compiled = self.compiled.clone();
        let schema = self.schema.clone();
        Ok(Box::new(stream.map(move |batch| {
            let batch = batch?;
            let arrays = compiled
                .iter()
                .map(|c| c.evaluate(&batch))
                .collect::<EvalResult<Vec<_>>>()?;
            Ok(RecordBatch::try_new(schema.clone(), arrays)?)
        })))
    }
}

// ---- limit -------------------------------------------------------------

pub struct LimitExec {
    input: Arc<dyn ExecutionPlan>,
    n: usize,
    schema: SchemaRef,
}

impl LimitExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, n: usize) -> Self {
        let schema = input.schema();
        Self { input, n, schema }
    }
}

struct LimitIter {
    inner: BatchStream,
    remaining: usize,
}

impl Iterator for LimitIter {
    type Item = EvalResult<RecordBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        match self.inner.next()? {
            Ok(batch) => {
                if batch.num_rows() <= self.remaining {
                    self.remaining -= batch.num_rows();
                    Some(Ok(batch))
                } else {
                    let trimmed = batch.slice(0, self.remaining);
                    self.remaining = 0;
                    Some(Ok(trimmed))
                }
            }
            Err(e) => Some(Err(e)),
        }
    }
}

impl ExecutionPlan for LimitExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        Ok(Box::new(LimitIter {
            inner: self.input.execute()?,
            remaining: self.n,
        }))
    }
}

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

use crate::eval::{evaluate, evaluate_predicate, EvalError, EvalResult};
use crate::exec::{BatchStream, ExecutionPlan};

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
    fn schema(&self) -> SchemaRef { self.schema.clone() }

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
    batch_size: usize,
}

impl ParquetScanExec {
    pub fn new(path: PathBuf, schema: SchemaRef) -> Self {
        Self { path, schema, projection: None, batch_size: DEFAULT_BATCH_SIZE }
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
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }
}

impl ExecutionPlan for ParquetScanExec {
    fn schema(&self) -> SchemaRef { self.schema.clone() }

    fn execute(&self) -> EvalResult<BatchStream> {
        let file = File::open(&self.path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_batch_size(self.batch_size);
        let reader = match &self.projection {
            None => builder.build()?,
            Some(cols) => {
                // Map column names to root-level indices in the parquet schema,
                // then build a ProjectionMask. Names that aren't in the file
                // surface as MissingColumn.
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
        Ok(Box::new(reader.map(|r| r.map_err(EvalError::from))))
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
    fn schema(&self) -> SchemaRef { self.schema.clone() }

    fn execute(&self) -> EvalResult<BatchStream> {
        // Hand back owned clones (RecordBatch is cheap to clone — Arc'd arrays).
        let owned = self.batches.clone();
        Ok(Box::new(owned.into_iter().map(Ok)))
    }
}

// ---- filter ------------------------------------------------------------

pub struct FilterExec {
    input: Arc<dyn ExecutionPlan>,
    predicate: Expr,
    schema: SchemaRef,
}

impl FilterExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, predicate: Expr) -> Self {
        let schema = input.schema();
        Self { input, predicate, schema }
    }
}

impl ExecutionPlan for FilterExec {
    fn schema(&self) -> SchemaRef { self.schema.clone() }

    fn execute(&self) -> EvalResult<BatchStream> {
        let stream = self.input.execute()?;
        let predicate = self.predicate.clone();
        Ok(Box::new(stream.map(move |batch| {
            let batch = batch?;
            let mask = evaluate_predicate(&predicate, &batch)?;
            let filtered = filter_record_batch(&batch, &mask)?;
            Ok(filtered)
        })))
    }
}

// ---- projection --------------------------------------------------------

pub struct ProjectionExec {
    input: Arc<dyn ExecutionPlan>,
    exprs: Vec<Expr>,
    schema: SchemaRef,
}

impl ProjectionExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, exprs: Vec<Expr>, schema: SchemaRef) -> Self {
        Self { input, exprs, schema }
    }
}

impl ExecutionPlan for ProjectionExec {
    fn schema(&self) -> SchemaRef { self.schema.clone() }

    fn execute(&self) -> EvalResult<BatchStream> {
        let stream = self.input.execute()?;
        let exprs = self.exprs.clone();
        let schema = self.schema.clone();
        Ok(Box::new(stream.map(move |batch| {
            let batch = batch?;
            let arrays = exprs
                .iter()
                .map(|e| evaluate(e, &batch))
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
    fn schema(&self) -> SchemaRef { self.schema.clone() }

    fn execute(&self) -> EvalResult<BatchStream> {
        Ok(Box::new(LimitIter { inner: self.input.execute()?, remaining: self.n }))
    }
}

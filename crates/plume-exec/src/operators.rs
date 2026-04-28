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
use plume_core::Expr;

use crate::eval::{evaluate, evaluate_predicate, EvalError, EvalResult};
use crate::exec::{BatchStream, ExecutionPlan};

const DEFAULT_BATCH_SIZE: usize = 8192;

// ---- CSV scan ----------------------------------------------------------

pub struct CsvScanExec {
    path: PathBuf,
    schema: SchemaRef,
    batch_size: usize,
    has_header: bool,
}

impl CsvScanExec {
    pub fn new(path: PathBuf, schema: SchemaRef) -> Self {
        Self { path, schema, batch_size: DEFAULT_BATCH_SIZE, has_header: true }
    }
}

impl ExecutionPlan for CsvScanExec {
    fn schema(&self) -> SchemaRef { self.schema.clone() }

    fn execute(&self) -> EvalResult<BatchStream> {
        let file = File::open(&self.path)?;
        let reader = csv::ReaderBuilder::new(self.schema.clone())
            .with_header(self.has_header)
            .with_batch_size(self.batch_size)
            .build(file)?;
        // arrow's CSV Reader is Iterator<Item = Result<RecordBatch, ArrowError>>.
        // Map the error type into ours.
        let mapped = reader.map(|r| r.map_err(EvalError::from));
        Ok(Box::new(mapped))
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

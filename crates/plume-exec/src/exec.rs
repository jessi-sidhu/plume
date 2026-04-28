//! `ExecutionPlan` trait — the physical-side counterpart to LogicalPlan.
//!
//! Operators own their inputs and stream `RecordBatch`es. The stream is
//! `'static + Send` so callers can hand it to a thread pool later (Week 13)
//! without lifetime gymnastics. Each `execute()` call yields a fresh stream;
//! operators never cache results between calls.

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;

use crate::eval::EvalResult;

pub type BatchStream = Box<dyn Iterator<Item = EvalResult<RecordBatch>> + Send>;

pub trait ExecutionPlan: Send + Sync {
    fn schema(&self) -> SchemaRef;
    fn execute(&self) -> EvalResult<BatchStream>;
}

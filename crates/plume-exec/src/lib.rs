//! plume-exec: physical execution.
//!
//! - `eval`: scalar expression evaluator (Week 3).
//! - `exec`: `ExecutionPlan` trait + batch stream (Week 4).
//! - `operators`: CsvScan, Memory, Filter, Projection, Limit (Week 4).
//! - `planner`: LogicalPlan → ExecutionPlan + `collect()` (Week 4).

pub mod eval;
pub mod exec;
pub mod operators;
pub mod planner;

pub use eval::{evaluate, evaluate_predicate, EvalError, EvalResult};
pub use exec::{BatchStream, ExecutionPlan};
pub use operators::{CsvScanExec, FilterExec, LimitExec, MemoryExec, ProjectionExec};
pub use planner::{collect, create_physical_plan};

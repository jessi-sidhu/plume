//! plume-exec: physical execution.
//!
//! - `eval`: scalar expression evaluator (Week 3).
//! - `exec`: `ExecutionPlan` trait + batch stream (Week 4).
//! - `operators`: CsvScan, Memory, Filter, Projection, Limit (Week 4).
//! - `aggregate`: Accumulator + global AggregateExec (Week 5).
//! - `planner`: LogicalPlan → ExecutionPlan + `collect()` (Week 4+).

pub mod aggregate;
pub mod compiled;
pub mod eval;
pub mod exec;
pub mod join;
pub mod keys;
pub mod operators;
pub mod parallel;
pub mod planner;
mod pruning;

pub use aggregate::{Accumulator, AggregateExec, HashAggregateExec};
pub use compiled::{compile, CompiledExpr};
pub use eval::{evaluate, evaluate_predicate, EvalError, EvalResult};
pub use exec::{BatchStream, ExecutionPlan};
pub use join::{CrossJoinExec, HashJoinExec};
pub use operators::{
    CsvScanExec, FilterExec, LimitExec, MemoryExec, ParquetScanExec, ProjectionExec,
};
pub use parallel::ParallelParquetScanExec;
pub use planner::{collect, create_physical_plan};

//! plume-exec: physical execution.
//!
//! - `eval`: scalar expression evaluator (Week 3).
//! - `exec`: `ExecutionPlan` trait + batch stream (Week 4).
//! - `operators`: CsvScan, Memory, Filter, Projection, Limit (Week 4).
//! - `aggregate`: Accumulator + global AggregateExec (Week 5).
//! - `sort`: SortExec (full sort) and TopKExec (bounded heap).
//! - `planner`: LogicalPlan → ExecutionPlan + `collect()` (Week 4+).

pub mod aggregate;
pub mod compiled;
pub mod eval;
pub mod exec;
pub mod join;
pub mod keys;
pub mod metrics;
pub mod operators;
pub mod parallel;
pub mod planner;
mod pruning;
pub mod sort;
pub mod topk_pruning;

pub use aggregate::{Accumulator, AggregateExec, HashAggregateExec};
pub use compiled::{compile, CompiledExpr};
pub use eval::{evaluate, evaluate_predicate, EvalError, EvalResult};
pub use exec::{BatchStream, ExecutionPlan};
pub use join::{CrossJoinExec, HashJoinExec};
pub use metrics::{ScanMetrics, ScanMetricsSnapshot};
pub use operators::{
    CsvScanExec, FilterExec, LimitExec, MemoryExec, ParquetScanExec, ProjectionExec,
};
pub use parallel::ParallelParquetScanExec;
pub use planner::{
    collect, collect_with_metrics, collect_with_options, create_physical_plan,
    create_physical_plan_with_metrics, create_physical_plan_with_options, PlanWithMetrics,
    PlannerOptions,
};
pub use sort::{SortExec, TopKExec};
pub use topk_pruning::{TopKBoundary, TopKPruning, TopKValue};

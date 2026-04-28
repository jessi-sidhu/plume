//! plume-exec: physical execution.
//!
//! Week 3 lands the expression evaluator. Operators (Filter/Project/Aggregate
//! exec nodes) follow in Week 4+.

pub mod eval;

pub use eval::{evaluate, evaluate_predicate, EvalError};

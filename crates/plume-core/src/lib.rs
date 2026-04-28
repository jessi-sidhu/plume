//! plume-core: LogicalPlan, Expr, DataFrame builder, schema propagation.
//!
//! No execution lives here — this crate is types and tree construction only.
//! Pure functions, deterministic, no I/O.

use thiserror::Error;

pub mod dataframe;
pub mod expr;
pub mod plan;

#[derive(Debug, Error)]
pub enum PlumeError {
    #[error("schema error: {0}")]
    Schema(String),
    #[error("type error: {0}")]
    Type(String),
    #[error("plan error: {0}")]
    Plan(String),
}

pub type Result<T> = std::result::Result<T, PlumeError>;

// Re-export the surface most callers want.
pub use dataframe::DataFrame;
pub use expr::{
    avg, col, count, count_star, lit, max, min, sum, AggregateFunction, BinaryOperator, Expr,
    ScalarValue,
};
pub use plan::{LogicalPlan, ScanSource};

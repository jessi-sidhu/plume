//! plume-core: LogicalPlan, Expr, Schema. Filled in starting Week 2.

use thiserror::Error;

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

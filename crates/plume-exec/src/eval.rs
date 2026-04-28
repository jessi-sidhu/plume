//! Expression evaluator. Walks an `Expr` against a `RecordBatch` and produces
//! an `ArrayRef`. Three-valued (Kleene) logic is delegated to Arrow's
//! `boolean::and_kleene` / `or_kleene`; comparison and arithmetic are
//! delegated to the Datum-based numeric/cmp kernels. We deliberately do not
//! write per-element loops — getting null semantics right by hand is exactly
//! the kind of thing this engine is supposed to *not* attempt.
//!
//! Coercion rule: strict. If a `BinaryOp` finds operands of mismatched types
//! the schema layer already rejected the plan; here, type mismatches surface
//! as Arrow errors. The evaluator does not silently widen Int32 to Int64 etc.
//!
//! Aggregate expressions are *not* evaluated here — they're driven by the
//! aggregate operator in Week 5+. Calling `evaluate` on an `Expr::Aggregate`
//! returns an error.

use std::sync::Arc;

use arrow::compute::kernels::{boolean, cmp, numeric};
use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::DataType;
use plume_core::{BinaryOperator, Expr, ScalarValue};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("column `{0}` not in batch schema")]
    MissingColumn(String),
    #[error("aggregate expressions are not evaluated here; use AggregateExec")]
    AggregateInScalarContext,
    #[error("untyped NULL literal cannot be evaluated standalone — wrap in a typed expression")]
    UntypedNullLiteral,
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("predicate is {0:?}, expected Boolean")]
    PredicateNotBoolean(DataType),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not implemented yet: {0}")]
    NotImplemented(String),
}

pub type EvalResult<T> = std::result::Result<T, EvalError>;

/// Evaluate `expr` against `batch`, returning an array of length `batch.num_rows()`.
pub fn evaluate(expr: &Expr, batch: &RecordBatch) -> EvalResult<ArrayRef> {
    match expr {
        Expr::Column(name) => batch
            .column_by_name(name)
            .cloned()
            .ok_or_else(|| EvalError::MissingColumn(name.clone())),
        Expr::Literal(v) => scalar_to_array(v, batch.num_rows()),
        Expr::BinaryOp { left, op, right } => {
            let l = evaluate(left, batch)?;
            let r = evaluate(right, batch)?;
            apply_binary(*op, &l, &r)
        }
        Expr::Aggregate { .. } => Err(EvalError::AggregateInScalarContext),
        Expr::Alias { expr, .. } => evaluate(expr, batch),
    }
}

/// Evaluate `expr` and require the result to be a `BooleanArray`. Caller-side
/// convenience for Filter operators.
pub fn evaluate_predicate(expr: &Expr, batch: &RecordBatch) -> EvalResult<BooleanArray> {
    let arr = evaluate(expr, batch)?;
    if arr.data_type() != &DataType::Boolean {
        return Err(EvalError::PredicateNotBoolean(arr.data_type().clone()));
    }
    Ok(arr.as_boolean().clone())
}

fn scalar_to_array(v: &ScalarValue, n: usize) -> EvalResult<ArrayRef> {
    Ok(match v {
        // A bare NULL literal has no type context — refuse rather than guess.
        // (count_star() carries Null as a placeholder arg, but is_aggregate
        // dispatch never reaches here.)
        ScalarValue::Null => return Err(EvalError::UntypedNullLiteral),
        ScalarValue::Boolean(b) => Arc::new(BooleanArray::from(vec![*b; n])),
        ScalarValue::Int32(v) => Arc::new(Int32Array::from(vec![*v; n])),
        ScalarValue::Int64(v) => Arc::new(Int64Array::from(vec![*v; n])),
        ScalarValue::Float64(v) => Arc::new(Float64Array::from(vec![*v; n])),
        ScalarValue::Utf8(s) => Arc::new(StringArray::from(vec![s.as_str(); n])),
    })
}

fn apply_binary(op: BinaryOperator, l: &ArrayRef, r: &ArrayRef) -> EvalResult<ArrayRef> {
    use BinaryOperator::*;
    Ok(match op {
        Eq => Arc::new(cmp::eq(l, r)?),
        NotEq => Arc::new(cmp::neq(l, r)?),
        Lt => Arc::new(cmp::lt(l, r)?),
        LtEq => Arc::new(cmp::lt_eq(l, r)?),
        Gt => Arc::new(cmp::gt(l, r)?),
        GtEq => Arc::new(cmp::gt_eq(l, r)?),
        // and_kleene / or_kleene are SQL three-valued logic. NEVER use the
        // plain `and`/`or` kernels here — those treat NULL as false.
        And => Arc::new(boolean::and_kleene(l.as_boolean(), r.as_boolean())?),
        Or => Arc::new(boolean::or_kleene(l.as_boolean(), r.as_boolean())?),
        Plus => numeric::add(l, r)?,
        Minus => numeric::sub(l, r)?,
        Multiply => numeric::mul(l, r)?,
        Divide => numeric::div(l, r)?,
    })
}

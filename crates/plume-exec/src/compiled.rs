//! Compiled (resolved) expressions — Week 11.
//!
//! `evaluate(&Expr, &RecordBatch)` re-resolves column names by HashMap lookup
//! on every batch and re-walks the whole expression tree. For a query that
//! processes 100+ batches, that's a meaningful per-batch overhead — name
//! lookups, redundant pattern matching, repeated literal materialization.
//!
//! `compile(&Expr, &Schema) -> CompiledExpr` does the resolution once at
//! operator construction time:
//!   - `Column(name)`  -> `Column { index, data_type }` resolved against the
//!     input schema. The hot path drops to `batch.column(index).clone()`.
//!   - `Literal(v)`    -> kept as a `ScalarValue`; the broadcast still has
//!     to size to the batch row count, but type dispatch happens once at
//!     compile time.
//!   - `BinaryOp`      -> children pre-compiled, op resolved upfront.
//!   - `Alias`         -> stripped (purely cosmetic at run time).
//!   - `Aggregate`     -> rejected; aggregates run inside AggregateExec.
//!
//! The tree-walking `evaluate` stays in `eval.rs` for tests and any caller
//! that doesn't have a schema in hand. Operators (FilterExec, ProjectionExec)
//! switch to the compiled path.

use std::sync::Arc;

use arrow::compute::kernels::{boolean, cmp, numeric};
use arrow::compute::{is_not_null, is_null};
use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Schema, TimeUnit};
use plume_core::{BinaryOperator, Expr, ScalarValue};

use crate::eval::{apply_not, EvalError, EvalResult};

/// `DataType` for `ScalarValue::TimestampMicros`, as a static so
/// `CompiledExpr::data_type` can hand out a reference.
static TIMESTAMP_MICROS: DataType = DataType::Timestamp(TimeUnit::Microsecond, None);

#[derive(Debug, Clone)]
pub enum CompiledExpr {
    Column {
        /// Pre-resolved index into the input batch's column vector.
        index: usize,
        data_type: DataType,
    },
    Literal {
        value: ScalarValue,
    },
    BinaryOp {
        left: Box<CompiledExpr>,
        op: BinaryOperator,
        right: Box<CompiledExpr>,
    },
    Not(Box<CompiledExpr>),
    IsNull(Box<CompiledExpr>),
    IsNotNull(Box<CompiledExpr>),
}

impl CompiledExpr {
    pub fn data_type(&self) -> &DataType {
        match self {
            CompiledExpr::Column { data_type, .. } => data_type,
            CompiledExpr::Literal { value } => match value {
                ScalarValue::Null => &DataType::Null,
                ScalarValue::Boolean(_) => &DataType::Boolean,
                ScalarValue::Int32(_) => &DataType::Int32,
                ScalarValue::Int64(_) => &DataType::Int64,
                ScalarValue::Float64(_) => &DataType::Float64,
                ScalarValue::Utf8(_) => &DataType::Utf8,
                ScalarValue::Date32(_) => &DataType::Date32,
                ScalarValue::TimestampMicros(_) => &TIMESTAMP_MICROS,
            },
            // BinaryOp output type is recoverable from operand types + op,
            // but operators that already cached an output schema don't need
            // this, so leave it for the caller via Expr::data_type when
            // they really need it.
            CompiledExpr::BinaryOp { .. } => &DataType::Null,
            CompiledExpr::Not(_) | CompiledExpr::IsNull(_) | CompiledExpr::IsNotNull(_) => {
                &DataType::Boolean
            }
        }
    }

    /// Hot path: produce the array for this expression on `batch`.
    pub fn evaluate(&self, batch: &RecordBatch) -> EvalResult<ArrayRef> {
        match self {
            CompiledExpr::Column { index, .. } => Ok(batch.column(*index).clone()),
            CompiledExpr::Literal { value } => scalar_to_array(value, batch.num_rows()),
            CompiledExpr::BinaryOp { left, op, right } => {
                let l = left.evaluate(batch)?;
                let r = right.evaluate(batch)?;
                apply_binary(*op, &l, &r)
            }
            CompiledExpr::Not(inner) => apply_not(&inner.evaluate(batch)?),
            CompiledExpr::IsNull(inner) => Ok(Arc::new(is_null(inner.evaluate(batch)?.as_ref())?)),
            CompiledExpr::IsNotNull(inner) => {
                Ok(Arc::new(is_not_null(inner.evaluate(batch)?.as_ref())?))
            }
        }
    }

    /// Predicate convenience: evaluate and unwrap to a `BooleanArray`.
    pub fn evaluate_predicate(&self, batch: &RecordBatch) -> EvalResult<BooleanArray> {
        let arr = self.evaluate(batch)?;
        if arr.data_type() != &DataType::Boolean {
            return Err(EvalError::PredicateNotBoolean(arr.data_type().clone()));
        }
        Ok(arr.as_boolean().clone())
    }
}

/// Resolve a logical `Expr` against an `input_schema`, producing a
/// `CompiledExpr` with column indices baked in. Errors at compile time
/// rather than per-batch.
pub fn compile(expr: &Expr, input_schema: &Schema) -> EvalResult<CompiledExpr> {
    match expr {
        Expr::Column(name) => {
            let index = input_schema
                .fields()
                .iter()
                .position(|f| f.name() == name)
                .ok_or_else(|| EvalError::MissingColumn(name.clone()))?;
            Ok(CompiledExpr::Column {
                index,
                data_type: input_schema.field(index).data_type().clone(),
            })
        }
        Expr::Literal(v) => Ok(CompiledExpr::Literal { value: v.clone() }),
        Expr::BinaryOp { left, op, right } => Ok(CompiledExpr::BinaryOp {
            left: Box::new(compile(left, input_schema)?),
            op: *op,
            right: Box::new(compile(right, input_schema)?),
        }),
        // Aggregates can't be evaluated row-by-row; they're driven by the
        // accumulator side of AggregateExec.
        Expr::Aggregate { .. } => Err(EvalError::AggregateInScalarContext),
        // Alias is cosmetic — strip and recurse.
        Expr::Alias { expr, .. } => compile(expr, input_schema),
        Expr::Not(inner) => Ok(CompiledExpr::Not(Box::new(compile(inner, input_schema)?))),
        Expr::IsNull(inner) => Ok(CompiledExpr::IsNull(Box::new(compile(
            inner,
            input_schema,
        )?))),
        Expr::IsNotNull(inner) => Ok(CompiledExpr::IsNotNull(Box::new(compile(
            inner,
            input_schema,
        )?))),
    }
}

fn scalar_to_array(v: &ScalarValue, n: usize) -> EvalResult<ArrayRef> {
    Ok(match v {
        ScalarValue::Null => return Err(EvalError::UntypedNullLiteral),
        ScalarValue::Boolean(b) => Arc::new(BooleanArray::from(vec![*b; n])),
        ScalarValue::Int32(v) => Arc::new(Int32Array::from(vec![*v; n])),
        ScalarValue::Int64(v) => Arc::new(Int64Array::from(vec![*v; n])),
        ScalarValue::Float64(v) => Arc::new(Float64Array::from(vec![*v; n])),
        ScalarValue::Utf8(s) => Arc::new(StringArray::from(vec![s.as_str(); n])),
        ScalarValue::Date32(d) => Arc::new(Date32Array::from(vec![*d; n])),
        ScalarValue::TimestampMicros(us) => Arc::new(TimestampMicrosecondArray::from(vec![*us; n])),
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
        And => Arc::new(boolean::and_kleene(l.as_boolean(), r.as_boolean())?),
        Or => Arc::new(boolean::or_kleene(l.as_boolean(), r.as_boolean())?),
        Plus => numeric::add(l, r)?,
        Minus => numeric::sub(l, r)?,
        Multiply => numeric::mul(l, r)?,
        Divide => numeric::div(l, r)?,
    })
}

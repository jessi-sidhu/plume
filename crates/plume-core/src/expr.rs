//! Expression tree + schema-time type resolution.
//!
//! `Expr` carries no values. It's a recipe for what to compute; actual eval
//! lives in `plume-exec` (Week 3+). What we *do* need at this layer is the
//! type system: given an input `Schema`, an `Expr` must be able to report the
//! output `DataType` and a sensible column name. Plan nodes lean on those two
//! methods to compute their own output schema (see `plan::LogicalPlan::schema`).

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::{PlumeError, Result};

/// A literal value carried inside an expression. Kept narrow on purpose —
/// extend as the test surface demands new types.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Null,
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    Float64(f64),
    Utf8(String),
}

impl ScalarValue {
    pub fn data_type(&self) -> DataType {
        match self {
            // Untyped NULL — the engine's coercion rules will pin it down at
            // the binary op that consumes it. For now it surfaces as Null.
            ScalarValue::Null => DataType::Null,
            ScalarValue::Boolean(_) => DataType::Boolean,
            ScalarValue::Int32(_) => DataType::Int32,
            ScalarValue::Int64(_) => DataType::Int64,
            ScalarValue::Float64(_) => DataType::Float64,
            ScalarValue::Utf8(_) => DataType::Utf8,
        }
    }

    /// Human-readable form used inside `Expr::display_name`. Strings are
    /// quoted; floats use minimal formatting.
    pub fn display_form(&self) -> String {
        match self {
            ScalarValue::Null => "NULL".to_string(),
            ScalarValue::Boolean(b) => b.to_string(),
            ScalarValue::Int32(v) => v.to_string(),
            ScalarValue::Int64(v) => v.to_string(),
            ScalarValue::Float64(v) => v.to_string(),
            ScalarValue::Utf8(s) => format!("\"{s}\""),
        }
    }
}

impl From<bool> for ScalarValue { fn from(v: bool) -> Self { ScalarValue::Boolean(v) } }
impl From<i32> for ScalarValue { fn from(v: i32) -> Self { ScalarValue::Int32(v) } }
impl From<i64> for ScalarValue { fn from(v: i64) -> Self { ScalarValue::Int64(v) } }
impl From<f64> for ScalarValue { fn from(v: f64) -> Self { ScalarValue::Float64(v) } }
impl From<&str> for ScalarValue { fn from(v: &str) -> Self { ScalarValue::Utf8(v.to_string()) } }
impl From<String> for ScalarValue { fn from(v: String) -> Self { ScalarValue::Utf8(v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Eq, NotEq, Lt, LtEq, Gt, GtEq,
    And, Or,
    Plus, Minus, Multiply, Divide,
}

impl BinaryOperator {
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Eq | Self::NotEq | Self::Lt | Self::LtEq | Self::Gt | Self::GtEq
        )
    }
    pub fn is_logical(self) -> bool {
        matches!(self, Self::And | Self::Or)
    }
    pub fn is_arithmetic(self) -> bool {
        matches!(self, Self::Plus | Self::Minus | Self::Multiply | Self::Divide)
    }
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "=", Self::NotEq => "!=",
            Self::Lt => "<", Self::LtEq => "<=", Self::Gt => ">", Self::GtEq => ">=",
            Self::And => "AND", Self::Or => "OR",
            Self::Plus => "+", Self::Minus => "-", Self::Multiply => "*", Self::Divide => "/",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    Count, CountStar, Sum, Avg, Min, Max,
}

impl AggregateFunction {
    pub fn name(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::CountStar => "count_star",
            Self::Sum => "sum",
            Self::Avg => "avg",
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(String),
    Literal(ScalarValue),
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    /// `arg` is unused (and conventionally `Literal(Null)`) when `fun` is
    /// `CountStar`. Kept on the same variant to keep pattern matching simple.
    Aggregate {
        fun: AggregateFunction,
        arg: Box<Expr>,
    },
    Alias {
        expr: Box<Expr>,
        name: String,
    },
}

impl Expr {
    /// Resolve this expression's output `DataType` against `input`. Returns
    /// `PlumeError::Schema` if a referenced column is missing, or
    /// `PlumeError::Type` if operand types don't match the operator's rules.
    pub fn data_type(&self, input: &Schema) -> Result<DataType> {
        match self {
            Expr::Column(name) => input
                .field_with_name(name)
                .map(|f| f.data_type().clone())
                .map_err(|_| PlumeError::Schema(format!("column `{name}` not in input schema"))),
            Expr::Literal(v) => Ok(v.data_type()),
            Expr::BinaryOp { left, op, right } => {
                let lt = left.data_type(input)?;
                let rt = right.data_type(input)?;
                resolve_binary(*op, &lt, &rt)
            }
            Expr::Aggregate { fun, arg } => {
                let inner = arg.data_type(input)?;
                resolve_aggregate(*fun, &inner)
            }
            Expr::Alias { expr, .. } => expr.data_type(input),
        }
    }

    /// Produce a stable display name for this expression — used as the column
    /// name in projection / aggregate output schemas.
    pub fn display_name(&self) -> String {
        match self {
            Expr::Column(name) => name.clone(),
            Expr::Literal(v) => v.display_form(),
            Expr::BinaryOp { left, op, right } => {
                format!("({} {} {})", left.display_name(), op.symbol(), right.display_name())
            }
            Expr::Aggregate { fun, arg } => match fun {
                AggregateFunction::CountStar => "count_star()".to_string(),
                _ => format!("{}({})", fun.name(), arg.display_name()),
            },
            Expr::Alias { name, .. } => name.clone(),
        }
    }

    pub fn nullable(&self, input: &Schema) -> Result<bool> {
        Ok(match self {
            Expr::Column(name) => input
                .field_with_name(name)
                .map(|f| f.is_nullable())
                .map_err(|_| PlumeError::Schema(format!("column `{name}` not in input schema")))?,
            Expr::Literal(v) => matches!(v, ScalarValue::Null),
            Expr::BinaryOp { left, right, .. } => {
                left.nullable(input)? || right.nullable(input)?
            }
            Expr::Aggregate { fun, .. } => match fun {
                // COUNT and COUNT(*) never return NULL.
                AggregateFunction::Count | AggregateFunction::CountStar => false,
                // Other aggregates over an empty/all-null group return NULL.
                _ => true,
            },
            Expr::Alias { expr, .. } => expr.nullable(input)?,
        })
    }

    /// Build the `Field` this expression contributes to an output schema.
    pub fn to_field(&self, input: &Schema) -> Result<Field> {
        Ok(Field::new(
            self.display_name(),
            self.data_type(input)?,
            self.nullable(input)?,
        ))
    }

    // ---- builder-style combinators ---------------------------------------
    pub fn alias(self, name: impl Into<String>) -> Expr {
        Expr::Alias { expr: Box::new(self), name: name.into() }
    }
    pub fn eq(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::Eq, rhs) }
    pub fn neq(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::NotEq, rhs) }
    pub fn lt(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::Lt, rhs) }
    pub fn lt_eq(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::LtEq, rhs) }
    pub fn gt(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::Gt, rhs) }
    pub fn gt_eq(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::GtEq, rhs) }
    pub fn and(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::And, rhs) }
    pub fn or(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::Or, rhs) }
    pub fn plus(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::Plus, rhs) }
    pub fn minus(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::Minus, rhs) }
    pub fn mul(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::Multiply, rhs) }
    pub fn div(self, rhs: Expr) -> Expr { binop(self, BinaryOperator::Divide, rhs) }
}

fn binop(l: Expr, op: BinaryOperator, r: Expr) -> Expr {
    Expr::BinaryOp { left: Box::new(l), op, right: Box::new(r) }
}

// ---- free-standing constructors -----------------------------------------

pub fn col(name: impl Into<String>) -> Expr { Expr::Column(name.into()) }
pub fn lit(v: impl Into<ScalarValue>) -> Expr { Expr::Literal(v.into()) }
pub fn count(arg: Expr) -> Expr { Expr::Aggregate { fun: AggregateFunction::Count, arg: Box::new(arg) } }
pub fn count_star() -> Expr {
    Expr::Aggregate { fun: AggregateFunction::CountStar, arg: Box::new(Expr::Literal(ScalarValue::Null)) }
}
pub fn sum(arg: Expr) -> Expr { Expr::Aggregate { fun: AggregateFunction::Sum, arg: Box::new(arg) } }
pub fn avg(arg: Expr) -> Expr { Expr::Aggregate { fun: AggregateFunction::Avg, arg: Box::new(arg) } }
pub fn min(arg: Expr) -> Expr { Expr::Aggregate { fun: AggregateFunction::Min, arg: Box::new(arg) } }
pub fn max(arg: Expr) -> Expr { Expr::Aggregate { fun: AggregateFunction::Max, arg: Box::new(arg) } }

// ---- type-resolution helpers --------------------------------------------

fn resolve_binary(op: BinaryOperator, l: &DataType, r: &DataType) -> Result<DataType> {
    if op.is_comparison() {
        // Comparable iff both sides match (or one is Null). Strict: no implicit
        // numeric widening at the schema layer — the eval layer will demand a
        // cast. Documented as the engine's coercion rule.
        if types_compatible(l, r) {
            Ok(DataType::Boolean)
        } else {
            Err(PlumeError::Type(format!(
                "comparison `{}` requires matching operand types, got {l:?} vs {r:?}",
                op.symbol()
            )))
        }
    } else if op.is_logical() {
        match (l, r) {
            (DataType::Boolean | DataType::Null, DataType::Boolean | DataType::Null) => {
                Ok(DataType::Boolean)
            }
            _ => Err(PlumeError::Type(format!(
                "logical `{}` requires Boolean operands, got {l:?} vs {r:?}",
                op.symbol()
            ))),
        }
    } else {
        // Arithmetic. Strict matching; no silent promotion. Null absorbs.
        match (l, r) {
            (DataType::Null, t) | (t, DataType::Null) if is_numeric(t) => Ok(t.clone()),
            (a, b) if a == b && is_numeric(a) => Ok(a.clone()),
            _ => Err(PlumeError::Type(format!(
                "arithmetic `{}` requires matching numeric operands, got {l:?} vs {r:?}",
                op.symbol()
            ))),
        }
    }
}

fn resolve_aggregate(fun: AggregateFunction, inner: &DataType) -> Result<DataType> {
    use AggregateFunction::*;
    Ok(match fun {
        Count | CountStar => DataType::Int64,
        Sum => match inner {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => DataType::Int64,
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => DataType::UInt64,
            DataType::Float32 | DataType::Float64 => DataType::Float64,
            other => return Err(PlumeError::Type(format!("sum() over non-numeric {other:?}"))),
        },
        Avg => {
            if !is_numeric(inner) {
                return Err(PlumeError::Type(format!("avg() over non-numeric {inner:?}")));
            }
            DataType::Float64
        }
        Min | Max => inner.clone(),
    })
}

fn is_numeric(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
            | DataType::Float32 | DataType::Float64
    )
}

fn types_compatible(l: &DataType, r: &DataType) -> bool {
    if l == r { return true; }
    matches!(l, DataType::Null) || matches!(r, DataType::Null)
}

/// Convenience: build a `SchemaRef` of fields produced by `exprs` against
/// `input`. Used by `Projection` and `Aggregate` plan nodes.
pub fn fields_for(input: &Schema, exprs: &[Expr]) -> Result<SchemaRef> {
    let fields: Result<Vec<Field>> = exprs.iter().map(|e| e.to_field(input)).collect();
    Ok(Arc::new(Schema::new(fields?)))
}

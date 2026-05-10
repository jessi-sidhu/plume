//! Hashable key representation shared by HashAggregateExec (GROUP BY) and
//! HashJoinExec (equi-join). Both operators do row-at-a-time probe-and-update
//! against a HashMap and need an `Eq + Hash` value built from one cell of an
//! Arrow array.
//!
//! Float keys hash by bit pattern, which means two NaNs with the same bit
//! pattern collapse into one key and otherwise form distinct ones. Documented;
//! matches what most engines do for f64 group / join keys.
//!
//! NULL handling: `extract_key` returns `KeyValue::Null` for null cells.
//! Callers decide what that means semantically — GROUP BY treats Null as a
//! valid group; equi-join treats any Null in a key vector as "no match"
//! (NULL ≠ NULL in SQL equi-join semantics).

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::ArrayRef;
use arrow_schema::DataType;

use crate::eval::{EvalError, EvalResult};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KeyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(u64), // f64 -> bits
    Str(String),
}

pub fn extract_key(arr: &ArrayRef, row: usize) -> EvalResult<KeyValue> {
    if arr.is_null(row) {
        return Ok(KeyValue::Null);
    }
    Ok(match arr.data_type() {
        DataType::Boolean => KeyValue::Bool(arr.as_boolean().value(row)),
        DataType::Int8 => {
            KeyValue::Int(arr.as_primitive::<arrow_array::types::Int8Type>().value(row) as i64)
        }
        DataType::Int16 => {
            KeyValue::Int(arr.as_primitive::<arrow_array::types::Int16Type>().value(row) as i64)
        }
        DataType::Int32 => KeyValue::Int(arr.as_primitive::<Int32Type>().value(row) as i64),
        DataType::Int64 => KeyValue::Int(arr.as_primitive::<Int64Type>().value(row)),
        DataType::Float32 => KeyValue::Float(
            (arr.as_primitive::<arrow_array::types::Float32Type>().value(row) as f64).to_bits(),
        ),
        DataType::Float64 => KeyValue::Float(arr.as_primitive::<Float64Type>().value(row).to_bits()),
        DataType::Utf8 => KeyValue::Str(arr.as_string::<i32>().value(row).to_string()),
        DataType::LargeUtf8 => KeyValue::Str(arr.as_string::<i64>().value(row).to_string()),
        other => return Err(EvalError::NotImplemented(format!("hash key over {other:?}"))),
    })
}

/// True if any element of `key` is `KeyValue::Null`. Equi-join uses this to
/// short-circuit: a row whose key contains any NULL never matches anything,
/// per SQL three-valued logic.
pub fn key_has_null(key: &[KeyValue]) -> bool {
    key.iter().any(|k| matches!(k, KeyValue::Null))
}

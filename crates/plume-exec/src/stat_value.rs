//! Values that can be compared with Parquet row-group min/max stats,
//! shared by top-k and join pruning.
//!
//! Only types whose Arrow representation and Parquet stats agree are
//! supported: Int32/Date32 (INT32 stats), Int64 and microsecond
//! timestamps (INT64 stats), Float64 (DOUBLE) and Utf8 (BYTE_ARRAY,
//! compared as bytes). Anything else yields `None`, and callers treat
//! `None` as "can't prune".

use std::cmp::Ordering;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{
    Date32Type, Float64Type, Int32Type, Int64Type, TimeUnit, TimestampMicrosecondType,
};
use arrow_schema::DataType;
use parquet::file::statistics::Statistics;

/// A column value that can be compared with Parquet row-group stats.
/// Integers, dates and microsecond timestamps all become `Int`.
#[derive(Debug, Clone, PartialEq)]
pub enum StatValue {
    Int(i64),
    Float(f64),
    Utf8(String),
}

impl StatValue {
    /// `None` when the variants differ (never happens for one column).
    pub(crate) fn compare(&self, other: &StatValue) -> Option<Ordering> {
        match (self, other) {
            (StatValue::Int(a), StatValue::Int(b)) => Some(a.cmp(b)),
            (StatValue::Float(a), StatValue::Float(b)) => Some(a.total_cmp(b)),
            (StatValue::Utf8(a), StatValue::Utf8(b)) => Some(a.as_bytes().cmp(b.as_bytes())),
            _ => None,
        }
    }

    /// Value at `row` of `array`, or `None` for NULL or an unsupported
    /// type.
    pub(crate) fn from_array(array: &ArrayRef, row: usize) -> Option<StatValue> {
        if array.is_null(row) {
            return None;
        }
        Some(match array.data_type() {
            DataType::Int32 => StatValue::Int(array.as_primitive::<Int32Type>().value(row) as i64),
            DataType::Int64 => StatValue::Int(array.as_primitive::<Int64Type>().value(row)),
            DataType::Date32 => {
                StatValue::Int(array.as_primitive::<Date32Type>().value(row) as i64)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                StatValue::Int(array.as_primitive::<TimestampMicrosecondType>().value(row))
            }
            DataType::Float64 => StatValue::Float(array.as_primitive::<Float64Type>().value(row)),
            DataType::Utf8 => StatValue::Utf8(array.as_string::<i32>().value(row).to_string()),
            _ => return None,
        })
    }
}

/// A column chunk's (min, max) stats as `StatValue`s, given the column's
/// Arrow type. `None` if the type is unsupported or doesn't match the
/// stats' physical type; either side is `None` when the writer left it
/// out (e.g. an all-NULL chunk). Inexact (truncated) stats are still
/// valid bounds: min is never above the real min, max never below.
pub(crate) fn column_bounds(
    stats: &Statistics,
    field_type: &DataType,
) -> Option<(Option<StatValue>, Option<StatValue>)> {
    Some(match (stats, field_type) {
        (Statistics::Int32(s), DataType::Int32 | DataType::Date32) => (
            s.min_opt().map(|&x| StatValue::Int(x as i64)),
            s.max_opt().map(|&x| StatValue::Int(x as i64)),
        ),
        (Statistics::Int64(s), DataType::Int64 | DataType::Timestamp(TimeUnit::Microsecond, _)) => {
            (
                s.min_opt().map(|&x| StatValue::Int(x)),
                s.max_opt().map(|&x| StatValue::Int(x)),
            )
        }
        (Statistics::Double(s), DataType::Float64) => (
            s.min_opt().map(|&x| StatValue::Float(x)),
            s.max_opt().map(|&x| StatValue::Float(x)),
        ),
        (Statistics::ByteArray(s), DataType::Utf8) => {
            let text = |b: &parquet::data_type::ByteArray| {
                std::str::from_utf8(b.data())
                    .ok()
                    .map(|x| StatValue::Utf8(x.to_string()))
            };
            (s.min_opt().and_then(text), s.max_opt().and_then(text))
        }
        _ => return None,
    })
}

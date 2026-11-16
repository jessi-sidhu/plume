//! Top-k pruning (Zimmerer et al., §5).
//!
//! A `TopKExec` whose first sort key is a plain scan column shares a
//! `TopKBoundary` with that scan. Once the heap holds k rows, TopK
//! publishes the first key of its current k-th best row. The scan reads
//! row groups one at a time, best possible value first, and before
//! opening each group checks whether every row in it is strictly worse
//! than the boundary on the first key. Such a row is strictly worse than
//! the k-th row overall, whatever the later keys say, and TopK only takes
//! rows strictly better than its k-th, so the group can't change the
//! answer. Ties are never skipped.
//!
//! What a group's "best possible value" is:
//!
//! - ASC: its min stat. DESC: its max stat. Parquet min/max are bounds
//!   even when inexact (truncated strings), which is all this needs.
//! - NULLS FIRST: a NULL beats every value, so a group that may hold
//!   NULLs (null count not known to be 0) is never skipped.
//! - NULLS LAST: NULLs are worse than any value, so they don't matter.
//! - Missing stats or an unsupported type: never skipped.
//!
//! A NULL boundary (the k-th row's key is NULL) publishes nothing, so
//! nothing is skipped. Floats compare with `total_cmp`, like Arrow's sort.
//! As with filter pruning, this assumes float data has no NaN: Parquet
//! leaves NaN out of min/max, so a DESC query over a group holding NaN
//! (which sorts above every number) would be pruned wrongly.

use std::cmp::Ordering;
use std::sync::{Arc, RwLock};

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{
    Date32Type, Float64Type, Int32Type, Int64Type, TimeUnit, TimestampMicrosecondType,
};
use arrow_schema::DataType;
use parquet::file::metadata::RowGroupMetaData;
use parquet::file::statistics::Statistics;

/// A sort-key value that can be compared with row-group stats. Integers,
/// dates and microsecond timestamps all become `Int`.
#[derive(Debug, Clone, PartialEq)]
pub enum TopKValue {
    Int(i64),
    Float(f64),
    Utf8(String),
}

impl TopKValue {
    /// `None` when the variants differ (never happens for one column).
    fn compare(&self, other: &TopKValue) -> Option<Ordering> {
        match (self, other) {
            (TopKValue::Int(a), TopKValue::Int(b)) => Some(a.cmp(b)),
            (TopKValue::Float(a), TopKValue::Float(b)) => Some(a.total_cmp(b)),
            (TopKValue::Utf8(a), TopKValue::Utf8(b)) => Some(a.as_bytes().cmp(b.as_bytes())),
            _ => None,
        }
    }

    /// Value at `row` of `array`, or `None` for NULL or an unsupported
    /// type.
    pub(crate) fn from_array(array: &ArrayRef, row: usize) -> Option<TopKValue> {
        if array.is_null(row) {
            return None;
        }
        Some(match array.data_type() {
            DataType::Int32 => TopKValue::Int(array.as_primitive::<Int32Type>().value(row) as i64),
            DataType::Int64 => TopKValue::Int(array.as_primitive::<Int64Type>().value(row)),
            DataType::Date32 => {
                TopKValue::Int(array.as_primitive::<Date32Type>().value(row) as i64)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                TopKValue::Int(array.as_primitive::<TimestampMicrosecondType>().value(row))
            }
            DataType::Float64 => TopKValue::Float(array.as_primitive::<Float64Type>().value(row)),
            DataType::Utf8 => TopKValue::Utf8(array.as_string::<i32>().value(row).to_string()),
            _ => return None,
        })
    }
}

/// The current k-th best first-key value, written by TopKExec and read by
/// the scan. It only ever gets better.
#[derive(Debug, Default)]
pub struct TopKBoundary {
    value: RwLock<Option<TopKValue>>,
}

impl TopKBoundary {
    pub fn get(&self) -> Option<TopKValue> {
        self.value.read().expect("boundary lock poisoned").clone()
    }
    pub(crate) fn set(&self, v: Option<TopKValue>) {
        *self.value.write().expect("boundary lock poisoned") = v;
    }
}

/// What a scan needs to prune for a TopK above it.
#[derive(Debug, Clone)]
pub struct TopKPruning {
    /// The first sort key: a column of the scan, by name.
    pub column: String,
    pub asc: bool,
    pub nulls_first: bool,
    pub boundary: Arc<TopKBoundary>,
}

impl TopKPruning {
    /// The best first-key value any row of `rg` could have, or `None` if
    /// the group can never be skipped (see the module docs). `field_type`
    /// is the column's Arrow type, which must agree with how its stats are
    /// stored.
    pub(crate) fn group_best(
        &self,
        rg: &RowGroupMetaData,
        col_idx: usize,
        field_type: &DataType,
    ) -> Option<TopKValue> {
        let stats = rg.column(col_idx).statistics()?;
        if self.nulls_first && stats.null_count_opt() != Some(0) {
            return None;
        }
        let asc = self.asc;
        match (stats, field_type) {
            (Statistics::Int32(s), DataType::Int32 | DataType::Date32) => {
                let v = if asc { s.min_opt() } else { s.max_opt() };
                v.map(|&x| TopKValue::Int(x as i64))
            }
            (
                Statistics::Int64(s),
                DataType::Int64 | DataType::Timestamp(TimeUnit::Microsecond, _),
            ) => {
                let v = if asc { s.min_opt() } else { s.max_opt() };
                v.map(|&x| TopKValue::Int(x))
            }
            (Statistics::Double(s), DataType::Float64) => {
                let v = if asc { s.min_opt() } else { s.max_opt() };
                v.map(|&x| TopKValue::Float(x))
            }
            (Statistics::ByteArray(s), DataType::Utf8) => {
                let v = if asc { s.min_opt() } else { s.max_opt() };
                v.and_then(|b| std::str::from_utf8(b.data()).ok())
                    .map(|x| TopKValue::Utf8(x.to_string()))
            }
            _ => None,
        }
    }

    /// True if a group whose best possible value is `best` can't hold a
    /// row the TopK would take, given the current boundary.
    pub(crate) fn can_skip(&self, best: &TopKValue) -> bool {
        let Some(boundary) = self.boundary.get() else {
            return false;
        };
        match best.compare(&boundary) {
            Some(Ordering::Greater) => self.asc,
            Some(Ordering::Less) => !self.asc,
            _ => false,
        }
    }

    /// Order `groups` for reading: groups that can never be skipped first
    /// (they may hold the best rows), then best possible value first.
    /// Equal values keep file order.
    pub(crate) fn read_order(&self, groups: &mut [(usize, Option<TopKValue>)]) {
        let asc = self.asc;
        groups.sort_by(|(ia, a), (ib, b)| {
            let by_value = match (a, b) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(x), Some(y)) => {
                    let o = x.compare(y).unwrap_or(Ordering::Equal);
                    if asc {
                        o
                    } else {
                        o.reverse()
                    }
                }
            };
            by_value.then(ia.cmp(ib))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pruning(asc: bool, boundary: Option<TopKValue>) -> TopKPruning {
        let b = Arc::new(TopKBoundary::default());
        b.set(boundary);
        TopKPruning {
            column: "x".into(),
            asc,
            nulls_first: false,
            boundary: b,
        }
    }

    #[test]
    fn skips_only_strictly_worse_groups() {
        // DESC with boundary 50: a group whose max is 49 can't help, a
        // group whose max is exactly 50 might tie and must be read.
        let p = pruning(false, Some(TopKValue::Int(50)));
        assert!(p.can_skip(&TopKValue::Int(49)));
        assert!(!p.can_skip(&TopKValue::Int(50)));
        assert!(!p.can_skip(&TopKValue::Int(51)));
        // ASC with boundary 50: min 51 is skippable, min 50 is not.
        let p = pruning(true, Some(TopKValue::Int(50)));
        assert!(p.can_skip(&TopKValue::Int(51)));
        assert!(!p.can_skip(&TopKValue::Int(50)));
        assert!(!p.can_skip(&TopKValue::Int(49)));
    }

    #[test]
    fn no_boundary_means_no_skipping() {
        let p = pruning(false, None);
        assert!(!p.can_skip(&TopKValue::Int(i64::MIN)));
    }

    #[test]
    fn floats_use_total_order() {
        let p = pruning(false, Some(TopKValue::Float(0.0)));
        assert!(p.can_skip(&TopKValue::Float(-0.5)));
        assert!(!p.can_skip(&TopKValue::Float(0.0)));
    }

    #[test]
    fn read_order_puts_unknown_groups_first_then_best_first() {
        let p = pruning(false, None);
        let mut g = vec![
            (0, Some(TopKValue::Int(10))),
            (1, None),
            (2, Some(TopKValue::Int(30))),
            (3, Some(TopKValue::Int(30))),
            (4, Some(TopKValue::Int(20))),
        ];
        p.read_order(&mut g);
        let order: Vec<usize> = g.iter().map(|(i, _)| *i).collect();
        assert_eq!(order, vec![1, 2, 3, 4, 0]);
    }
}

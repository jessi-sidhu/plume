//! Join pruning (Zimmerer et al., §6).
//!
//! `HashJoinExec` drains its build side (the right input) before it calls
//! `execute()` on the probe side (the left input). So by the time the
//! probe-side Parquet scan plans its reads, the build-side key values are
//! known. For each join key that is a plain probe-scan column, the join
//! publishes the build keys' [min, max] (NULLs ignored) through a shared
//! `JoinFilter`, and the scan skips row groups whose stats for that column
//! fall entirely outside it. This is pruning only: the join still does the
//! exact matching on every row it reads.
//!
//! It is only correct for join types that drop probe rows without a
//! match: Inner, Semi and Right. Left, Full and Anti keep unmatched probe
//! rows, so the planner never wires a filter for them.
//!
//! If the build side has no non-NULL key at all, no probe row can match,
//! and the filter says `Empty`: the scan skips every row group. A row
//! group whose probe key is all NULL is skipped too, since NULL keys never
//! match.
//!
//! Float keys aren't summarized (`Unknown`): join equality and the stats'
//! ordering disagree on NaN and -0.0, and float join keys are rare.

use std::sync::{Arc, RwLock};

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{Date32Type, Int32Type, Int64Type, TimeUnit, TimestampMicrosecondType};
use arrow_schema::DataType;
use parquet::file::metadata::RowGroupMetaData;

use crate::stat_value::{column_bounds, StatValue};

/// What the build side's keys for one join column look like.
#[derive(Debug, Clone, PartialEq)]
pub enum JoinKeyRange {
    /// No non-NULL key: no probe row can match.
    Empty,
    /// Every non-NULL build key lies in [min, max].
    Range { min: StatValue, max: StatValue },
    /// Unsupported type: prune nothing.
    Unknown,
}

/// Shared between a HashJoinExec (writer) and the probe-side scan
/// (reader). `None` until the build side has been drained.
#[derive(Debug, Default)]
pub struct JoinFilter {
    state: RwLock<Option<JoinKeyRange>>,
}

impl JoinFilter {
    pub fn get(&self) -> Option<JoinKeyRange> {
        self.state
            .read()
            .expect("join filter lock poisoned")
            .clone()
    }
    pub(crate) fn set(&self, range: JoinKeyRange) {
        *self.state.write().expect("join filter lock poisoned") = Some(range);
    }
}

/// One join-key filter as the probe scan sees it.
#[derive(Debug, Clone)]
pub struct JoinPruning {
    /// The probe key: a column of the scan, by name.
    pub column: String,
    pub filter: Arc<JoinFilter>,
}

impl JoinPruning {
    /// True if no row of `rg` can have a key that matches the build side.
    pub(crate) fn can_skip(&self, rg: &RowGroupMetaData, col_idx: usize, ty: &DataType) -> bool {
        let range = match self.filter.get() {
            None | Some(JoinKeyRange::Unknown) => return false,
            Some(JoinKeyRange::Empty) => return true,
            Some(JoinKeyRange::Range { min, max }) => (min, max),
        };
        let Some(stats) = rg.column(col_idx).statistics() else {
            return false;
        };
        if stats.null_count_opt() == Some(rg.num_rows() as u64) {
            return true;
        }
        let Some((Some(lo), Some(hi))) = column_bounds(stats, ty) else {
            return false;
        };
        use std::cmp::Ordering::{Greater, Less};
        hi.compare(&range.0) == Some(Less) || lo.compare(&range.1) == Some(Greater)
    }
}

/// Summarize a build-side key array, ignoring NULLs.
pub(crate) fn key_range(array: &ArrayRef) -> JoinKeyRange {
    fn ints(values: impl Iterator<Item = Option<i64>>) -> JoinKeyRange {
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        let mut any = false;
        for v in values.flatten() {
            any = true;
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if any {
            JoinKeyRange::Range {
                min: StatValue::Int(lo),
                max: StatValue::Int(hi),
            }
        } else {
            JoinKeyRange::Empty
        }
    }
    match array.data_type() {
        DataType::Int32 => ints(
            array
                .as_primitive::<Int32Type>()
                .iter()
                .map(|v| v.map(i64::from)),
        ),
        DataType::Int64 => ints(array.as_primitive::<Int64Type>().iter()),
        DataType::Date32 => ints(
            array
                .as_primitive::<Date32Type>()
                .iter()
                .map(|v| v.map(i64::from)),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            ints(array.as_primitive::<TimestampMicrosecondType>().iter())
        }
        DataType::Utf8 => {
            let strings = array.as_string::<i32>();
            let mut lo: Option<&str> = None;
            let mut hi: Option<&str> = None;
            for s in strings.iter().flatten() {
                if lo.map_or(true, |l| s.as_bytes() < l.as_bytes()) {
                    lo = Some(s);
                }
                if hi.map_or(true, |h| s.as_bytes() > h.as_bytes()) {
                    hi = Some(s);
                }
            }
            match (lo, hi) {
                (Some(l), Some(h)) => JoinKeyRange::Range {
                    min: StatValue::Utf8(l.to_string()),
                    max: StatValue::Utf8(h.to_string()),
                },
                _ => JoinKeyRange::Empty,
            }
        }
        _ => JoinKeyRange::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, StringArray};

    use super::*;

    #[test]
    fn range_ignores_nulls() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![Some(5), None, Some(-2), Some(9)]));
        assert_eq!(
            key_range(&a),
            JoinKeyRange::Range {
                min: StatValue::Int(-2),
                max: StatValue::Int(9)
            }
        );
    }

    #[test]
    fn all_null_or_empty_build_is_empty() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![None, None]));
        assert_eq!(key_range(&a), JoinKeyRange::Empty);
        let a: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        assert_eq!(key_range(&a), JoinKeyRange::Empty);
    }

    #[test]
    fn strings_compare_as_bytes() {
        let a: ArrayRef = Arc::new(StringArray::from(vec!["pear", "apple", "zebra"]));
        assert_eq!(
            key_range(&a),
            JoinKeyRange::Range {
                min: StatValue::Utf8("apple".into()),
                max: StatValue::Utf8("zebra".into())
            }
        );
    }

    #[test]
    fn floats_are_not_summarized() {
        let a: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        assert_eq!(key_range(&a), JoinKeyRange::Unknown);
    }
}

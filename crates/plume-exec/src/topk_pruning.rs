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

use arrow_schema::DataType;
use parquet::file::metadata::RowGroupMetaData;

use crate::stat_value::{column_bounds, StatValue};

/// The current k-th best first-key value, written by TopKExec and read by
/// the scan. It only ever gets better.
#[derive(Debug, Default)]
pub struct TopKBoundary {
    value: RwLock<Option<StatValue>>,
}

impl TopKBoundary {
    pub fn get(&self) -> Option<StatValue> {
        self.value.read().expect("boundary lock poisoned").clone()
    }
    pub(crate) fn set(&self, v: Option<StatValue>) {
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
    ) -> Option<StatValue> {
        let stats = rg.column(col_idx).statistics()?;
        if self.nulls_first && stats.null_count_opt() != Some(0) {
            return None;
        }
        let (min, max) = column_bounds(stats, field_type)?;
        if self.asc {
            min
        } else {
            max
        }
    }

    /// True if a group whose best possible value is `best` can't hold a
    /// row the TopK would take, given the current boundary.
    pub(crate) fn can_skip(&self, best: &StatValue) -> bool {
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
    pub(crate) fn read_order(&self, groups: &mut [(usize, Option<StatValue>)]) {
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

    fn pruning(asc: bool, boundary: Option<StatValue>) -> TopKPruning {
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
        let p = pruning(false, Some(StatValue::Int(50)));
        assert!(p.can_skip(&StatValue::Int(49)));
        assert!(!p.can_skip(&StatValue::Int(50)));
        assert!(!p.can_skip(&StatValue::Int(51)));
        // ASC with boundary 50: min 51 is skippable, min 50 is not.
        let p = pruning(true, Some(StatValue::Int(50)));
        assert!(p.can_skip(&StatValue::Int(51)));
        assert!(!p.can_skip(&StatValue::Int(50)));
        assert!(!p.can_skip(&StatValue::Int(49)));
    }

    #[test]
    fn no_boundary_means_no_skipping() {
        let p = pruning(false, None);
        assert!(!p.can_skip(&StatValue::Int(i64::MIN)));
    }

    #[test]
    fn floats_use_total_order() {
        let p = pruning(false, Some(StatValue::Float(0.0)));
        assert!(p.can_skip(&StatValue::Float(-0.5)));
        assert!(!p.can_skip(&StatValue::Float(0.0)));
    }

    #[test]
    fn read_order_puts_unknown_groups_first_then_best_first() {
        let p = pruning(false, None);
        let mut g = vec![
            (0, Some(StatValue::Int(10))),
            (1, None),
            (2, Some(StatValue::Int(30))),
            (3, Some(StatValue::Int(30))),
            (4, Some(StatValue::Int(20))),
        ];
        p.read_order(&mut g);
        let order: Vec<usize> = g.iter().map(|(i, _)| *i).collect();
        assert_eq!(order, vec![1, 2, 3, 4, 0]);
    }
}

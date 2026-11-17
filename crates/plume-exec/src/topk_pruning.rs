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
//! nothing is skipped.
//!
//! Stats initialization (§5.4): before reading anything, the scan can
//! set a boundary from FullMatch groups, whose rows are all known to
//! reach the TopK. For DESC, a FullMatch group with n non-NULL sort
//! values and min stat m guarantees n rows with key >= m. Taking groups
//! by min, best first, until their non-NULL rows add up to k gives a
//! value b with at least k rows >= b, so the true k-th best is >= b and
//! b is a safe boundary (ASC mirrors this with max). It's only used when
//! every Filter between the TopK and the scan is covered by the scan's
//! pushdown predicates, and not under NULLS FIRST. Once set, TopK only
//! ever tightens the boundary, so a heap that is still filling with
//! worse rows can't loosen it. Floats compare with `total_cmp`, like Arrow's sort.
//! As with filter pruning, this assumes float data has no NaN: Parquet
//! leaves NaN out of min/max, so a DESC query over a group holding NaN
//! (which sorts above every number) would be pruned wrongly.

use std::cmp::Ordering;
use std::sync::{Arc, RwLock};

use arrow_schema::DataType;
use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};

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

    /// Replace the boundary with `v` only if `v` is better. Under NULLS
    /// FIRST, where no stats boundary is ever set and the k-th value can
    /// legitimately become NULL, this is a plain `set`.
    pub(crate) fn tighten(&self, v: Option<StatValue>, asc: bool, nulls_first: bool) {
        if nulls_first {
            return self.set(v);
        }
        let mut cur = self.value.write().expect("boundary lock poisoned");
        let better = match (&*cur, &v) {
            (_, None) => false,
            (None, Some(_)) => true,
            (Some(c), Some(n)) => match n.compare(c) {
                Some(Ordering::Less) => asc,
                Some(Ordering::Greater) => !asc,
                _ => false,
            },
        };
        if better {
            *cur = v;
        }
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
    /// The TopK's k, for stats initialization.
    pub k: usize,
    /// Whether the scan may set the boundary from FullMatch stats before
    /// reading. The planner sets it only when every Filter in between is
    /// covered by the scan's pushdown predicates.
    pub init_from_stats: bool,
}

impl TopKPruning {
    /// A boundary implied by FullMatch groups alone, or `None`. `full`
    /// lists the FullMatch groups the scan will read.
    pub(crate) fn stats_boundary(
        &self,
        metadata: &ParquetMetaData,
        full: &[usize],
        col_idx: usize,
        field_type: &DataType,
    ) -> Option<StatValue> {
        if !self.init_from_stats || self.nulls_first || self.k == 0 {
            return None;
        }
        // (guaranteed bound, non-NULL rows) per group: every non-NULL value
        // is >= min (DESC) or <= max (ASC).
        let mut groups: Vec<(StatValue, u64)> = full
            .iter()
            .filter_map(|&g| {
                let rg = metadata.row_group(g);
                let stats = rg.column(col_idx).statistics()?;
                let nulls = stats.null_count_opt()?;
                let (min, max) = column_bounds(stats, field_type)?;
                let bound = if self.asc { max? } else { min? };
                Some((bound, (rg.num_rows().max(0) as u64).saturating_sub(nulls)))
            })
            .collect();
        let asc = self.asc;
        groups.sort_by(|(a, _), (b, _)| {
            let o = a.compare(b).unwrap_or(Ordering::Equal);
            if asc {
                o
            } else {
                o.reverse()
            }
        });
        let mut rows = 0u64;
        for (bound, n) in groups {
            rows += n;
            if rows >= self.k as u64 {
                return Some(bound);
            }
        }
        None
    }

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
            k: 1,
            init_from_stats: false,
        }
    }

    /// Groups of 10 rows: [100,110), [90,100), [2,10) plus two NULLs,
    /// [50,60). Returns the file's metadata.
    fn stats_file() -> (tempfile::TempDir, std::sync::Arc<ParquetMetaData>) {
        use arrow::array::Int64Array;
        use arrow_schema::{Field, Schema};
        let mut x: Vec<Option<i64>> = (100..110)
            .chain(90..100)
            .chain(0..10)
            .chain(50..60)
            .map(Some)
            .collect();
        x[20] = None;
        x[21] = None;
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(x))],
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.parquet");
        let props = parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_size(10)
            .build();
        let mut w = parquet::arrow::ArrowWriter::try_new(
            std::fs::File::create(&path).unwrap(),
            schema,
            Some(props),
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let b = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            std::fs::File::open(&path).unwrap(),
        )
        .unwrap();
        (dir, b.metadata().clone())
    }

    fn init(asc: bool, k: usize) -> TopKPruning {
        TopKPruning {
            k,
            init_from_stats: true,
            ..pruning(asc, None)
        }
    }

    #[test]
    fn stats_boundary_counts_guaranteed_rows() {
        let (_dir, md) = stats_file();
        let all = [0, 1, 2, 3];
        let b =
            |p: TopKPruning, groups: &[usize]| p.stats_boundary(&md, groups, 0, &DataType::Int64);
        // DESC, by min: 100 (10 rows), 90 (10), 50 (10), 2 (8 non-NULL).
        assert_eq!(b(init(false, 5), &all), Some(StatValue::Int(100)));
        assert_eq!(b(init(false, 20), &all), Some(StatValue::Int(90)));
        assert_eq!(b(init(false, 21), &all), Some(StatValue::Int(50)));
        assert_eq!(b(init(false, 38), &all), Some(StatValue::Int(2)));
        // 38 non-NULL rows in total: k = 39 can't be guaranteed.
        assert_eq!(b(init(false, 39), &all), None);
        // ASC, by max: 9 (8 non-NULL rows), 59 (10), ...
        assert_eq!(b(init(true, 8), &all), Some(StatValue::Int(9)));
        assert_eq!(b(init(true, 9), &all), Some(StatValue::Int(59)));
        // Only the groups passed in count.
        assert_eq!(b(init(false, 15), &[0]), None);
    }

    #[test]
    fn stats_boundary_only_when_allowed() {
        let (_dir, md) = stats_file();
        let b = |p: TopKPruning| p.stats_boundary(&md, &[0, 1], 0, &DataType::Int64);
        assert_eq!(
            b(TopKPruning {
                init_from_stats: false,
                ..init(false, 5)
            }),
            None
        );
        assert_eq!(
            b(TopKPruning {
                nulls_first: true,
                ..init(false, 5)
            }),
            None
        );
        assert_eq!(b(init(false, 0)), None);
    }

    #[test]
    fn tighten_never_loosens() {
        let p = pruning(false, Some(StatValue::Int(50)));
        p.boundary.tighten(Some(StatValue::Int(40)), false, false);
        assert_eq!(p.boundary.get(), Some(StatValue::Int(50)));
        p.boundary.tighten(None, false, false);
        assert_eq!(p.boundary.get(), Some(StatValue::Int(50)));
        p.boundary.tighten(Some(StatValue::Int(60)), false, false);
        assert_eq!(p.boundary.get(), Some(StatValue::Int(60)));
        // ASC: smaller is better.
        let p = pruning(true, Some(StatValue::Int(50)));
        p.boundary.tighten(Some(StatValue::Int(60)), true, false);
        assert_eq!(p.boundary.get(), Some(StatValue::Int(50)));
        p.boundary.tighten(Some(StatValue::Int(40)), true, false);
        assert_eq!(p.boundary.get(), Some(StatValue::Int(40)));
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

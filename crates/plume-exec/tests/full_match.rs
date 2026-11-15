//! FullMatch row groups pass through the Filter without being evaluated.
//!
//! The sales.parquet tests use the gen_fixtures file: 8 row groups of 125K
//! rows, non-NULL prices in bands [0,125), [125,250), ... [875,1000). With
//! `price > 600` the bands from 625 up (3 groups) are fully inside the
//! predicate, [500,625) straddles it, and the rest can't match. They skip
//! if the fixture is missing; the temp-file tests always run.
//!
//! Plans are built by hand (scan with pushdown predicates, Filter on top)
//! and run through the planner, which is what decides whether the Filter
//! may skip FullMatch batches.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_core::{col, lit, Expr, LogicalPlan};
use plume_exec::{collect_with_metrics, ScanMetricsSnapshot};

fn sales_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures/sales.parquet");
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIP: {} missing, run gen_fixtures first", p.display());
        None
    }
}

/// Filter(predicate) over a Parquet scan with `pushdown` attached, the
/// shape PredicatePushdown produces.
fn filtered_scan(path: &Path, pushdown: Vec<Expr>, predicate: Expr) -> LogicalPlan {
    let schema = plume_io::read_parquet_schema(path).unwrap();
    let scan = LogicalPlan::scan_parquet(path.to_path_buf(), schema);
    let scan = if pushdown.is_empty() {
        scan
    } else {
        LogicalPlan::with_pushdown_predicates(scan, pushdown).unwrap()
    };
    LogicalPlan::filter(scan, predicate).unwrap()
}

fn run(plan: &LogicalPlan) -> (RecordBatch, ScanMetricsSnapshot) {
    let (batches, metrics) = collect_with_metrics(plan).unwrap();
    let schema = plan.schema();
    (concat_batches(&schema, &batches).unwrap(), metrics[0])
}

#[test]
fn full_match_groups_skip_filter_evaluation() {
    let Some(path) = sales_path() else { return };
    let pred = col("price").gt(lit(600.0));

    let (with_pushdown, m) = run(&filtered_scan(&path, vec![pred.clone()], pred.clone()));
    // Reference: no pushdown, so no pruning and no skipping.
    let (reference, m_ref) = run(&filtered_scan(&path, vec![], pred));

    assert_eq!(with_pushdown, reference, "results must not change");
    // No pushdown predicates: every group is trivially FullMatch for the
    // scan, but the filter isn't covered, so it evaluates everything.
    assert_eq!(m_ref.row_groups_full_match, 8);
    assert_eq!(m_ref.rows_filter_skipped, 0);

    assert_eq!(m.row_groups_total, 8);
    assert_eq!(m.row_groups_pruned, 4);
    assert_eq!(m.row_groups_full_match, 3);
    assert_eq!(m.rows_filter_skipped, 3 * 125_000);
}

#[test]
fn no_group_is_full_when_the_only_survivor_straddles_the_bound() {
    // price > 900 keeps only [875,1000), which straddles 900.
    let Some(path) = sales_path() else { return };
    let pred = col("price").gt(lit(900.0));
    let (with_pushdown, m) = run(&filtered_scan(&path, vec![pred.clone()], pred.clone()));
    let (reference, _) = run(&filtered_scan(&path, vec![], pred));
    assert_eq!(with_pushdown, reference);
    assert_eq!(m.row_groups_pruned, 7);
    assert_eq!(m.row_groups_full_match, 0);
    assert_eq!(m.rows_filter_skipped, 0);
}

#[test]
fn must_not_skip_when_filter_has_a_conjunct_the_scan_never_saw() {
    // The scan only knows `price > 600`, so it still calls three groups
    // FullMatch. But `price * 2 < 1500` (price < 750) rejects most of
    // those rows, so the Filter has to evaluate every batch.
    let Some(path) = sales_path() else { return };
    let pushed = col("price").gt(lit(600.0));
    let full_pred = pushed
        .clone()
        .and(col("price").mul(lit(2.0)).lt(lit(1500.0)));

    let (with_pushdown, m) = run(&filtered_scan(&path, vec![pushed], full_pred.clone()));
    let (reference, _) = run(&filtered_scan(&path, vec![], full_pred));

    assert_eq!(with_pushdown, reference);
    assert_eq!(m.row_groups_full_match, 3);
    assert_eq!(m.rows_filter_skipped, 0, "filter must not skip");
    assert!(with_pushdown.num_rows() < 3 * 125_000);
}

#[test]
fn literal_on_the_left_is_still_covered() {
    // The optimizer pushes `600 < price` as `price > 600`; coverage has
    // to see through the flip.
    let Some(path) = sales_path() else { return };
    let (with_pushdown, m) = run(&filtered_scan(
        &path,
        vec![col("price").gt(lit(600.0))],
        lit(600.0).lt(col("price")),
    ));
    let (reference, _) = run(&filtered_scan(&path, vec![], lit(600.0).lt(col("price"))));
    assert_eq!(with_pushdown, reference);
    assert_eq!(m.rows_filter_skipped, 3 * 125_000);
}

/// Write `values` to a temp Parquet file as one nullable Int64 column `x`,
/// `rows_per_group` rows per row group.
fn write_x(dir: &Path, values: Vec<Option<i64>>, rows_per_group: usize) -> PathBuf {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))]).unwrap();
    let path = dir.join("x.parquet");
    let props = WriterProperties::builder()
        .set_max_row_group_size(rows_per_group)
        .build();
    let mut w = ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

#[test]
fn must_not_skip_a_group_containing_nulls() {
    // Group 0: 10..=19 with a NULL; group 1: 20..=29, no NULLs. For
    // x > 5 only group 1 is FullMatch: skipping the filter on group 0
    // would let the NULL row through.
    let dir = tempfile::tempdir().unwrap();
    let mut values: Vec<Option<i64>> = (10..20).map(Some).collect();
    values[3] = None;
    values.extend((20..30).map(Some));
    let path = write_x(dir.path(), values, 10);

    let pred = col("x").gt(lit(5i64));
    let (out, m) = run(&filtered_scan(&path, vec![pred.clone()], pred));
    assert_eq!(m.row_groups_full_match, 1);
    assert_eq!(m.rows_filter_skipped, 10);
    assert_eq!(out.num_rows(), 19, "the NULL row must be filtered out");
    assert_eq!(out.column(0).null_count(), 0);
}

#[test]
fn full_and_partial_groups_interleave_in_file_order() {
    // Full, partial, full: batches must come back in file order with the
    // right rows, whichever reader produced them.
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<Option<i64>> = (100..110).chain(0..10).chain(200..210).map(Some).collect();
    let path = write_x(dir.path(), values, 10);

    let pred = col("x").gt_eq(lit(5i64));
    let (out, m) = run(&filtered_scan(&path, vec![pred.clone()], pred));
    assert_eq!(m.row_groups_full_match, 2);
    assert_eq!(m.rows_filter_skipped, 20);
    let got: Vec<i64> = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    let want: Vec<i64> = (100..110).chain(5..10).chain(200..210).collect();
    assert_eq!(got, want);
}

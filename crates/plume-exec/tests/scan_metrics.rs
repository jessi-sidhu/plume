//! ScanMetrics on ParquetScanExec and ParallelParquetScanExec.
//!
//! The sales.parquet tests use the gen_fixtures file (8 row groups of
//! 125K rows, price bands [0,125), [125,250), ... [875,1000)), so
//! `price > 950` can only match the last group. They skip if the fixture
//! is missing; the temp-file test always runs.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_core::{col, lit, Expr, LogicalPlan};
use plume_exec::{
    collect_with_metrics, ExecutionPlan, ParallelParquetScanExec, ParquetScanExec, ScanMetrics,
    ScanMetricsSnapshot,
};

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
        eprintln!("SKIP: {} missing — run gen_fixtures first", p.display());
        None
    }
}

fn drain(plan: Arc<dyn ExecutionPlan>) -> usize {
    plan.execute().unwrap().map(|b| b.unwrap().num_rows()).sum()
}

fn price_gt_950() -> Vec<Expr> {
    vec![col("price").gt(lit(950.0_f64))]
}

fn run_sequential(path: &Path, preds: Vec<Expr>) -> (usize, ScanMetricsSnapshot) {
    let schema = plume_io::read_parquet_schema(path).unwrap();
    let scan = ParquetScanExec::new(path.to_path_buf(), schema).with_pushdown_predicates(preds);
    let metrics: Arc<ScanMetrics> = scan.metrics();
    let rows = drain(Arc::new(scan));
    (rows, metrics.snapshot())
}

fn run_parallel(path: &Path, preds: Vec<Expr>) -> (usize, ScanMetricsSnapshot) {
    let schema = plume_io::read_parquet_schema(path).unwrap();
    let scan =
        ParallelParquetScanExec::new(path.to_path_buf(), schema).with_pushdown_predicates(preds);
    let metrics = scan.metrics();
    let rows = drain(Arc::new(scan));
    (rows, metrics.snapshot())
}

#[test]
fn sequential_sales_price_gt_950_prunes_7_of_8() {
    let Some(p) = sales_path() else { return };
    let (rows, m) = run_sequential(&p, price_gt_950());
    assert_eq!(m.row_groups_total, 8);
    assert_eq!(m.row_groups_pruned, 7);
    assert!(m.bytes_read < m.bytes_total, "{m:?}");
    assert!(m.bytes_read > 0, "{m:?}");
    assert_eq!(m.rows_decoded, 125_000);
    assert_eq!(rows as u64, m.rows_decoded);
}

#[test]
fn sequential_sales_no_predicate_prunes_nothing() {
    let Some(p) = sales_path() else { return };
    let (rows, m) = run_sequential(&p, vec![]);
    assert_eq!(m.row_groups_total, 8);
    assert_eq!(m.row_groups_pruned, 0);
    assert_eq!(m.bytes_read, m.bytes_total);
    assert_eq!(m.rows_decoded, 1_000_000);
    assert_eq!(rows as u64, m.rows_decoded);
}

#[test]
fn parallel_sales_price_gt_950_prunes_7_of_8() {
    let Some(p) = sales_path() else { return };
    let (rows, m) = run_parallel(&p, price_gt_950());
    assert_eq!(m.row_groups_total, 8);
    assert_eq!(m.row_groups_pruned, 7);
    assert!(m.bytes_read < m.bytes_total, "{m:?}");
    assert_eq!(m.rows_decoded, 125_000);
    assert_eq!(rows as u64, m.rows_decoded);
}

#[test]
fn parallel_sales_no_predicate_prunes_nothing() {
    let Some(p) = sales_path() else { return };
    let (_, m) = run_parallel(&p, vec![]);
    assert_eq!(m.row_groups_total, 8);
    assert_eq!(m.row_groups_pruned, 0);
    assert_eq!(m.bytes_read, m.bytes_total);
    assert_eq!(m.rows_decoded, 1_000_000);
}

#[test]
fn sequential_and_parallel_agree() {
    let Some(p) = sales_path() else { return };
    assert_eq!(
        run_sequential(&p, price_gt_950()).1,
        run_parallel(&p, price_gt_950()).1
    );
}

#[test]
fn projection_only_counts_projected_columns() {
    let Some(p) = sales_path() else { return };
    let full = plume_io::read_parquet_schema(&p).unwrap();
    let price_only: SchemaRef = Arc::new(Schema::new(vec![full
        .field_with_name("price")
        .unwrap()
        .clone()]));
    let scan = ParquetScanExec::with_projection(p.clone(), price_only, vec!["price".into()]);
    let m = scan.metrics();
    drain(Arc::new(scan));

    let (_, all_cols) = run_sequential(&p, vec![]);
    assert!(m.bytes_total() > 0);
    assert!(m.bytes_total() < all_cols.bytes_total);
    assert_eq!(m.bytes_read(), m.bytes_total());
}

#[test]
fn metrics_reset_on_each_execute() {
    let Some(p) = sales_path() else { return };
    let schema = plume_io::read_parquet_schema(&p).unwrap();
    let concrete = ParquetScanExec::new(p, schema).with_pushdown_predicates(price_gt_950());
    let m = concrete.metrics();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(concrete);
    drain(plan.clone());
    let first = m.snapshot();
    drain(plan);
    assert_eq!(m.snapshot(), first);
    assert_eq!(first.rows_decoded, 125_000);
}

#[test]
fn collect_with_metrics_reports_planner_scans() {
    let Some(p) = sales_path() else { return };
    let scan = plume_io::scan_parquet_auto(&p).unwrap().into_plan();
    let plan = LogicalPlan::with_pushdown_predicates(scan, price_gt_950()).unwrap();
    let (batches, metrics) = collect_with_metrics(&plan).unwrap();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].row_groups_total, 8);
    assert_eq!(metrics[0].row_groups_pruned, 7);
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows as u64, metrics[0].rows_decoded);
}

// ---- self-contained: no fixture needed -----------------------------------

/// 4 row groups of 100 rows, `id` sorted 0..400.
fn write_sorted_file(path: &Path) {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_size(100)
        .build();
    let mut w =
        ArrowWriter::try_new(File::create(path).unwrap(), schema.clone(), Some(props)).unwrap();
    let ids: Vec<i64> = (0..400).collect();
    let vs: Vec<f64> = (0..400).map(|i| i as f64).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Float64Array::from(vs)),
        ],
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

#[test]
fn temp_file_point_lookup_prunes_3_of_4() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("sorted.parquet");
    write_sorted_file(&path);

    for (rows, m) in [
        run_sequential(&path, vec![col("id").eq(lit(250_i64))]),
        run_parallel(&path, vec![col("id").eq(lit(250_i64))]),
    ] {
        assert_eq!(m.row_groups_total, 4);
        assert_eq!(m.row_groups_pruned, 3);
        assert_eq!(m.rows_decoded, 100);
        assert_eq!(rows, 100);
        assert!(m.bytes_read < m.bytes_total, "{m:?}");
    }

    let (_, m) = run_sequential(&path, vec![]);
    assert_eq!(m.row_groups_pruned, 0);
    assert_eq!(m.bytes_read, m.bytes_total);
    assert_eq!(m.rows_decoded, 400);
}

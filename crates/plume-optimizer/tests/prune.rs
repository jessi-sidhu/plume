//! Tests for the column-pruning optimizer pass.
//!
//! These tests inspect the rewritten LogicalPlan structure (no execution).
//! End-to-end correctness — pruned plan produces same results as unpruned —
//! lives in plume-bench/tests/prune_diff.rs.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{avg, col, count_star, lit, sum, DataFrame, LogicalPlan};
use plume_optimizer::prune_columns;

fn sales_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
    ]))
}

fn scan_projection(plan: &LogicalPlan) -> Option<Vec<String>> {
    match plan {
        LogicalPlan::Scan { projection, .. } => projection.clone(),
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Projection { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Limit { input, .. } => scan_projection(input),
    }
}

#[test]
fn projection_alone_prunes_unreferenced_columns() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .project(vec![col("category"), col("price")])
        .unwrap()
        .into_plan();
    let pruned = prune_columns(plan).unwrap();
    let proj = scan_projection(&pruned).expect("scan must be projected");
    assert_eq!(proj, vec!["category", "price"]);
}

#[test]
fn filter_then_projection_keeps_filter_columns() {
    // Filter references `region`; projection emits `category, price`.
    // The scan must read all three.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("region").eq(lit("west"))).unwrap()
        .project(vec![col("category"), col("price")]).unwrap()
        .into_plan();
    let pruned = prune_columns(plan).unwrap();
    let proj = scan_projection(&pruned).expect("scan must be projected");
    // File-schema order is preserved — region comes before price.
    assert_eq!(proj, vec!["category", "region", "price"]);
}

#[test]
fn aggregate_collects_group_and_agg_arg_columns() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .aggregate(
            vec![col("region")],
            vec![avg(col("price")).alias("avg_price"), sum(col("quantity")).alias("total_qty")],
        )
        .unwrap()
        .into_plan();
    let pruned = prune_columns(plan).unwrap();
    let proj = scan_projection(&pruned).expect("scan must be projected");
    assert_eq!(proj, vec!["region", "price", "quantity"]);
}

#[test]
fn count_star_only_keeps_one_column_for_row_count() {
    // COUNT(*) doesn't reference any input column. We still need *some*
    // column to count rows; the optimizer keeps the first.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .aggregate(vec![], vec![count_star().alias("n")])
        .unwrap()
        .into_plan();
    let pruned = prune_columns(plan).unwrap();
    let proj = scan_projection(&pruned).expect("scan must be projected");
    assert_eq!(proj, vec!["order_id"]);
}

#[test]
fn pruning_is_a_noop_when_every_column_is_required() {
    // Project all five columns explicitly — the optimizer should leave
    // projection as None (signal: nothing to prune).
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .project(vec![
            col("order_id"), col("category"), col("region"), col("price"), col("quantity"),
        ])
        .unwrap()
        .into_plan();
    let pruned = prune_columns(plan).unwrap();
    assert!(scan_projection(&pruned).is_none());
}

#[test]
fn csv_scans_are_intentionally_not_pruned() {
    // CSV is row-oriented; pruning only saves parse CPU, not I/O. The
    // optimizer skips CSV scans entirely.
    let plan = DataFrame::scan_csv("p.csv", sales_schema())
        .project(vec![col("price")])
        .unwrap()
        .into_plan();
    let pruned = prune_columns(plan).unwrap();
    assert!(scan_projection(&pruned).is_none(),
        "CSV scans must not be pruned (W8 design choice)");
}

#[test]
fn limit_is_transparent_to_pruning() {
    // Projection on top is what triggers pruning. Limit must not widen
    // the required-column set passing through it.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(100.0_f64))).unwrap()
        .limit(10)
        .project(vec![col("category")]).unwrap()
        .into_plan();
    let pruned = prune_columns(plan).unwrap();
    let proj = scan_projection(&pruned).expect("scan must be projected");
    // Projection wants `category`; filter wants `price`. Limit adds nothing.
    assert_eq!(proj.len(), 2);
    assert!(proj.contains(&"category".to_string()));
    assert!(proj.contains(&"price".to_string()));
}

#[test]
fn nested_aggregate_after_filter_keeps_filter_and_agg_cols() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("region").eq(lit("east"))).unwrap()
        .aggregate(
            vec![col("category")],
            vec![sum(col("quantity")).alias("total_qty")],
        )
        .unwrap()
        .into_plan();
    let pruned = prune_columns(plan).unwrap();
    let proj = scan_projection(&pruned).expect("scan must be projected");
    // Need: category (group), quantity (agg arg), region (filter).
    assert_eq!(proj.len(), 3);
    assert!(proj.contains(&"category".to_string()));
    assert!(proj.contains(&"region".to_string()));
    assert!(proj.contains(&"quantity".to_string()));
}

#[test]
fn explain_shows_projection_when_optimizer_runs() {
    let plan = DataFrame::scan_parquet("s.parquet", sales_schema())
        .aggregate(vec![col("region")], vec![avg(col("price")).alias("avg_price")])
        .unwrap()
        .into_plan();
    let pruned = prune_columns(plan).unwrap();
    let s = pruned.explain();
    assert!(s.contains("projection=["), "explain should show projection list:\n{s}");
    assert!(s.contains("region"));
    assert!(s.contains("price"));
}

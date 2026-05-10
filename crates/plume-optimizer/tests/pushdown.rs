//! Tests for the predicate-pushdown optimizer pass.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{col, count_star, lit, sum, DataFrame, LogicalPlan};
use plume_optimizer::{optimize, pushdown_predicates};

fn sales_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
    ]))
}

fn scan_pushdowns(plan: &LogicalPlan) -> Option<Vec<String>> {
    match plan {
        LogicalPlan::Scan { pushdown_predicates, .. } => {
            Some(pushdown_predicates.iter().map(|p| p.display_name()).collect())
        }
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Projection { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Limit { input, .. } => scan_pushdowns(input),
        LogicalPlan::Join { left, .. } => scan_pushdowns(left),
    }
}

#[test]
fn single_comparison_is_pushed_to_parquet_scan() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(500.0_f64))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    let preds = scan_pushdowns(&optimized).unwrap();
    assert_eq!(preds, vec!["(price > 500)"]);
}

#[test]
fn conjunction_decomposes_into_separate_pushdowns() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(100.0_f64)).and(col("region").eq(lit("west")))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    let preds = scan_pushdowns(&optimized).unwrap();
    assert_eq!(preds.len(), 2);
    assert!(preds.contains(&"(price > 100)".to_string()));
    assert!(preds.contains(&"(region = \"west\")".to_string()));
}

#[test]
fn literal_on_left_gets_swapped_with_flipped_op() {
    // 500 < price  ==  price > 500
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(lit(500.0_f64).lt(col("price"))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    let preds = scan_pushdowns(&optimized).unwrap();
    assert_eq!(preds, vec!["(price > 500)"]);
}

#[test]
fn or_predicate_does_not_get_pushed() {
    // OR is not decomposable into independent row-group filters.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(900.0_f64)).or(col("price").lt(lit(100.0_f64)))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    let preds = scan_pushdowns(&optimized).unwrap();
    assert!(preds.is_empty(), "OR clauses should not be pushed: {preds:?}");
}

#[test]
fn mixed_conjunction_keeps_only_pushable_parts() {
    // (price > 100) AND (region = category): the second clause is col-vs-col,
    // not col-vs-literal, so it stays in the Filter.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(
            col("price").gt(lit(100.0_f64))
                .and(col("region").eq(col("category")))
        ).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    let preds = scan_pushdowns(&optimized).unwrap();
    assert_eq!(preds, vec!["(price > 100)"]);
}

#[test]
fn pushdown_does_not_fire_for_csv_scans() {
    let plan = DataFrame::scan_csv("p.csv", sales_schema())
        .filter(col("price").gt(lit(100.0_f64))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    let preds = scan_pushdowns(&optimized).unwrap();
    assert!(preds.is_empty(), "CSV scan must not receive pushdowns: {preds:?}");
}

#[test]
fn filter_remains_in_plan_after_pushdown() {
    // Pushdown is a hint, not a replacement: the Filter that fed the
    // predicates must still be there to do the per-row check.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(500.0_f64))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    assert!(matches!(optimized, LogicalPlan::Filter { .. }));
}

#[test]
fn pushdown_works_through_projection_to_underlying_filter() {
    // SELECT col FROM ... WHERE pred LIMIT N  with projection above the
    // filter — the optimizer should still find the filter and push.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(700.0_f64))).unwrap()
        .project(vec![col("order_id"), col("price")]).unwrap()
        .limit(50)
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    let preds = scan_pushdowns(&optimized).unwrap();
    assert_eq!(preds, vec!["(price > 700)"]);
}

#[test]
fn optimize_runs_pushdown_then_pruning_in_one_call() {
    // After optimize(), the scan should have BOTH pushdown predicates AND
    // a column projection (since GROUP BY only needs price + quantity).
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(500.0_f64))).unwrap()
        .aggregate(vec![], vec![sum(col("quantity")).alias("total_qty"), count_star().alias("n")]).unwrap()
        .into_plan();
    let optimized = optimize(plan).unwrap();
    let preds = scan_pushdowns(&optimized).unwrap();
    assert_eq!(preds, vec!["(price > 500)"]);

    // Walk to the scan and check projection.
    fn scan_proj(p: &LogicalPlan) -> Option<Vec<String>> {
        match p {
            LogicalPlan::Scan { projection, .. } => projection.clone(),
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Limit { input, .. } => scan_proj(input),
            LogicalPlan::Join { left, .. } => scan_proj(left),
        }
    }
    let proj = scan_proj(&optimized).expect("scan should be projected");
    // Need price (filter), quantity (sum). order_id stays out.
    assert!(proj.contains(&"price".to_string()));
    assert!(proj.contains(&"quantity".to_string()));
    assert!(!proj.contains(&"order_id".to_string()));
}

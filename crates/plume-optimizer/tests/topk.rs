//! SortLimitToTopK, and how the other rules treat Sort and TopK.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{col, count_star, lit, DataFrame, LogicalPlan};
use plume_optimizer::{Optimizer, OptimizerRule, SortLimitToTopK};

fn sales_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("cost", DataType::Float64, false),
    ]))
}

fn sales() -> DataFrame {
    DataFrame::scan_parquet("sales.parquet", sales_schema())
}

fn optimize(df: DataFrame) -> LogicalPlan {
    Optimizer::default().optimize(df.into_plan()).unwrap()
}

/// Node names top-down along the first input, e.g. ["TopK", "Scan"].
fn spine(plan: &LogicalPlan) -> Vec<&'static str> {
    let (name, next): (&str, Option<&LogicalPlan>) = match plan {
        LogicalPlan::Scan { .. } => ("Scan", None),
        LogicalPlan::Filter { input, .. } => ("Filter", Some(input)),
        LogicalPlan::Projection { input, .. } => ("Projection", Some(input)),
        LogicalPlan::Aggregate { input, .. } => ("Aggregate", Some(input)),
        LogicalPlan::Limit { input, .. } => ("Limit", Some(input)),
        LogicalPlan::Sort { input, .. } => ("Sort", Some(input)),
        LogicalPlan::TopK { input, .. } => ("TopK", Some(input)),
        LogicalPlan::Join { left, .. } => ("Join", Some(left)),
    };
    let mut out = vec![name];
    if let Some(n) = next {
        out.extend(spine(n));
    }
    out
}

fn find_scan(plan: &LogicalPlan) -> &LogicalPlan {
    match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Projection { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::TopK { input, .. } => find_scan(input),
        LogicalPlan::Join { left, .. } => find_scan(left),
    }
}

fn top_k_of(plan: &LogicalPlan) -> Option<usize> {
    match plan {
        LogicalPlan::TopK { k, .. } => Some(*k),
        LogicalPlan::Scan { .. } => None,
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Projection { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. } => top_k_of(input),
        LogicalPlan::Join { left, .. } => top_k_of(left),
    }
}

#[test]
fn rule_has_a_unique_name() {
    assert_eq!(SortLimitToTopK.name(), "sort_limit_to_topk");
}

#[test]
fn limit_over_sort_becomes_top_k() {
    let plan = optimize(sales().sort(vec![col("price").desc()]).unwrap().limit(10));
    assert_eq!(spine(&plan), vec!["TopK", "Scan"]);
    assert_eq!(top_k_of(&plan), Some(10));
    assert!(plan
        .explain()
        .contains("TopK: k=10 [price DESC NULLS LAST]"));
}

#[test]
fn rewrites_through_a_projection() {
    let df = sales()
        .sort(vec![col("price").asc()])
        .unwrap()
        .project(vec![col("order_id"), col("price")])
        .unwrap()
        .limit(5);
    // The rule alone: the full optimizer would also drop the projection
    // once column pruning makes it an identity.
    let plan = SortLimitToTopK.rewrite(df.into_plan()).unwrap();
    assert_eq!(spine(&plan), vec!["Projection", "TopK", "Scan"]);
}

#[test]
fn nested_limits_keep_the_smaller_k() {
    let df = sales()
        .sort(vec![col("price").asc()])
        .unwrap()
        .limit(100)
        .limit(7);
    assert_eq!(top_k_of(&optimize(df)), Some(7));
    let df = sales()
        .sort(vec![col("price").asc()])
        .unwrap()
        .limit(3)
        .limit(50);
    assert_eq!(top_k_of(&optimize(df)), Some(3));
}

#[test]
fn optimizer_is_idempotent() {
    let df = sales()
        .filter(col("price").gt(lit(10.0)))
        .unwrap()
        .sort(vec![col("cost").desc()])
        .unwrap()
        .limit(10);
    let once = optimize(df);
    let twice = Optimizer::default().optimize(once.clone()).unwrap();
    assert_eq!(once, twice);
}

#[test]
fn column_pruning_keeps_the_sort_keys() {
    // Only order_id is projected, but the scan must still read price.
    let df = sales()
        .sort(vec![col("price").desc()])
        .unwrap()
        .project(vec![col("order_id")])
        .unwrap()
        .limit(10);
    let plan = optimize(df);
    let LogicalPlan::Scan { projection, .. } = find_scan(&plan) else {
        unreachable!()
    };
    let mut cols = projection.clone().expect("scan should be pruned");
    cols.sort();
    assert_eq!(cols, vec!["order_id", "price"]);
}

// ---- must not fire ------------------------------------------------------

#[test]
fn must_not_rewrite_across_a_filter() {
    // Limit -> Filter -> Sort: the first 10 rows *after* the filter, which
    // a TopK below the filter would not produce.
    let df = sales()
        .sort(vec![col("price").desc()])
        .unwrap()
        .filter(col("region").eq(lit("west")))
        .unwrap()
        .limit(10);
    let plan = SortLimitToTopK.rewrite(df.into_plan()).unwrap();
    assert_eq!(spine(&plan), vec!["Limit", "Filter", "Sort", "Scan"]);
}

#[test]
fn must_not_rewrite_across_an_aggregate() {
    let df = sales()
        .sort(vec![col("price").desc()])
        .unwrap()
        .aggregate(vec![col("region")], vec![count_star().alias("n")])
        .unwrap()
        .limit(3);
    let plan = optimize(df);
    assert_eq!(top_k_of(&plan), None, "{}", plan.explain());
}

#[test]
fn must_not_fold_a_limit_into_the_scan_under_a_sort() {
    // LimitPushdown capping the scan at 10 rows would sort the wrong 10.
    let df = sales().sort(vec![col("price").desc()]).unwrap().limit(10);
    let plan = optimize(df);
    let LogicalPlan::Scan {
        limit, limit_hint, ..
    } = find_scan(&plan)
    else {
        unreachable!()
    };
    assert_eq!(*limit, None);
    assert_eq!(*limit_hint, None);
}

#[test]
fn must_not_push_a_filter_below_a_top_k() {
    // Filter over TopK keeps "the best 10, then those in the west"; moving
    // the filter below would return up to 10 western rows instead.
    let df = sales()
        .sort(vec![col("price").desc()])
        .unwrap()
        .limit(10)
        .filter(col("region").eq(lit("west")))
        .unwrap();
    let plan = optimize(df);
    assert_eq!(spine(&plan), vec!["Filter", "TopK", "Scan"]);
    let LogicalPlan::Scan {
        pushdown_predicates,
        ..
    } = find_scan(&plan)
    else {
        unreachable!()
    };
    assert!(pushdown_predicates.is_empty());
}

#[test]
fn a_sort_without_a_limit_stays_a_sort() {
    let plan = optimize(sales().sort(vec![col("price").desc()]).unwrap());
    assert_eq!(spine(&plan), vec!["Sort", "Scan"]);
}

#[test]
fn sort_rejects_aggregates_and_unknown_columns() {
    assert!(sales().sort(vec![count_star().asc()]).is_err());
    assert!(sales().sort(vec![col("nope").asc()]).is_err());
    assert!(sales().sort(vec![]).is_err());
}

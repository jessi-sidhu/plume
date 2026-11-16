//! LimitPruning: when it attaches a limit hint to the scan, and every
//! shape where it must not.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{col, count_star, lit, DataFrame, JoinType, LogicalPlan};
use plume_optimizer::{LimitPruning, Optimizer, OptimizerRule};

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

/// Every scan's limit hint, depth-first (left before right).
fn hints(plan: &LogicalPlan) -> Vec<Option<usize>> {
    match plan {
        LogicalPlan::Scan { limit_hint, .. } => vec![*limit_hint],
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Projection { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::TopK { input, .. } => hints(input),
        LogicalPlan::Join { left, right, .. } => {
            let mut h = hints(left);
            h.extend(hints(right));
            h
        }
    }
}

fn optimized_hints(df: DataFrame) -> Vec<Option<usize>> {
    hints(&Optimizer::default().optimize(df.into_plan()).unwrap())
}

#[test]
fn rule_has_a_unique_name() {
    assert_eq!(LimitPruning.name(), "limit_pruning");
}

#[test]
fn hints_a_filtered_scan() {
    let df = sales()
        .filter(col("price").gt(lit(600.0)))
        .unwrap()
        .limit(100);
    assert_eq!(optimized_hints(df), vec![Some(100)]);
}

#[test]
fn hints_through_a_projection_above_the_filter() {
    let df = sales()
        .filter(
            col("price")
                .gt(lit(600.0))
                .and(col("region").eq(lit("west"))),
        )
        .unwrap()
        .project(vec![col("order_id"), col("price")])
        .unwrap()
        .limit(10);
    assert_eq!(optimized_hints(df), vec![Some(10)]);
}

#[test]
fn hints_with_the_literal_on_the_left() {
    let df = sales()
        .filter(lit(600.0).lt(col("price")))
        .unwrap()
        .limit(5);
    assert_eq!(optimized_hints(df), vec![Some(5)]);
}

#[test]
fn nested_limits_keep_the_tighter_hint() {
    let df = sales()
        .filter(col("price").gt(lit(600.0)))
        .unwrap()
        .limit(100)
        .limit(7);
    let plan = Optimizer::default().optimize(df.into_plan()).unwrap();
    assert_eq!(hints(&plan), vec![Some(7)]);
}

#[test]
fn optimizer_is_idempotent_with_the_hint() {
    let df = sales()
        .filter(col("price").gt(lit(600.0)))
        .unwrap()
        .limit(100);
    let once = Optimizer::default().optimize(df.into_plan()).unwrap();
    let twice = Optimizer::default().optimize(once.clone()).unwrap();
    assert_eq!(once, twice);
}

// ---- must not fire ------------------------------------------------------

#[test]
fn must_not_hint_when_a_conjunct_is_not_pushed_down() {
    // `price > cost` compares two columns, so the scan can't classify it.
    // A FullMatch group (judged on price > 600 alone) could still fail it,
    // and reading "just enough" FullMatch rows would return too few.
    let df = sales()
        .filter(
            col("price")
                .gt(lit(600.0))
                .and(col("price").gt(col("cost"))),
        )
        .unwrap()
        .limit(100);
    assert_eq!(optimized_hints(df), vec![None]);
}

#[test]
fn must_not_hint_through_an_aggregate() {
    let df = sales()
        .filter(col("price").gt(lit(600.0)))
        .unwrap()
        .aggregate(vec![col("region")], vec![count_star().alias("n")])
        .unwrap()
        .limit(3);
    assert_eq!(optimized_hints(df), vec![None]);
}

#[test]
fn must_not_hint_through_a_join() {
    let other = DataFrame::scan_parquet(
        "regions.parquet",
        Arc::new(Schema::new(vec![Field::new(
            "region_name",
            DataType::Utf8,
            false,
        )])),
    );
    let df = sales()
        .filter(col("price").gt(lit(600.0)))
        .unwrap()
        .join(
            other,
            vec![(col("region"), col("region_name"))],
            JoinType::Inner,
        )
        .unwrap()
        .limit(10);
    assert_eq!(optimized_hints(df), vec![None, None]);
}

#[test]
fn must_not_hint_a_filter_below_a_projection() {
    // Filter -> Projection -> Scan: the filter's column names aren't
    // guaranteed to be the scan's, so the rule stays out.
    let df = sales()
        .project(vec![col("price"), col("order_id")])
        .unwrap()
        .filter(col("price").gt(lit(600.0)))
        .unwrap()
        .limit(10);
    let plan = df.into_plan();
    // Run only this rule so nothing else reshapes the plan first.
    let out = LimitPruning.rewrite(plan).unwrap();
    assert_eq!(hints(&out), vec![None]);
}

#[test]
fn must_not_hint_without_a_limit() {
    let df = sales().filter(col("price").gt(lit(600.0))).unwrap();
    assert_eq!(optimized_hints(df), vec![None]);
}

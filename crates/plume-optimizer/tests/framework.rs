//! Tests for the OptimizerRule trait, the fixed-point driver, and the
//! three new W10 rules: ConstantFolding, DeadProjectionElim, LimitPushdown.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{
    col, count_star, lit, not, sum, BinaryOperator, DataFrame, Expr, LogicalPlan, ScalarValue,
};
use plume_optimizer::{
    explain_optimized, ColumnPruning, ConstantFolding, DeadProjectionElim, LimitPushdown,
    Optimizer, OptimizerRule, PredicatePushdown,
};

fn sales_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
    ]))
}

// Helpers for plan introspection.
fn scan_node(plan: &LogicalPlan) -> &LogicalPlan {
    match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Projection { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::TopK { input, .. } => scan_node(input),
        LogicalPlan::Join { left, .. } => scan_node(left),
    }
}

// =======================================================================
// Trait + driver
// =======================================================================

#[test]
fn rule_names_are_unique_and_descriptive() {
    let names = vec![
        ConstantFolding.name(),
        DeadProjectionElim.name(),
        PredicatePushdown.name(),
        ColumnPruning.name(),
        LimitPushdown.name(),
    ];
    let unique: std::collections::HashSet<_> = names.iter().copied().collect();
    assert_eq!(
        names.len(),
        unique.len(),
        "rule names must be unique: {names:?}"
    );
}

#[test]
fn fixed_point_driver_terminates_on_idempotent_rules() {
    // Two passes of the default optimizer over a non-trivial plan must
    // converge — second call should produce a structurally identical plan.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(500.0_f64)))
        .unwrap()
        .aggregate(
            vec![col("region")],
            vec![sum(col("quantity")).alias("total_qty")],
        )
        .unwrap()
        .into_plan();
    let opt = Optimizer::default();
    let once = opt.optimize(plan).unwrap();
    let twice = opt.optimize(once.clone()).unwrap();
    assert_eq!(once.explain(), twice.explain());
}

#[test]
fn explain_optimized_renders_before_after() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(500.0_f64)))
        .unwrap()
        .into_plan();
    let optimized = Optimizer::default().optimize(plan.clone()).unwrap();
    let s = explain_optimized(&plan, &optimized);
    assert!(s.contains("BEFORE:"));
    assert!(s.contains("AFTER:"));
    assert!(s.contains("pushdown=["));
}

// =======================================================================
// ConstantFolding
// =======================================================================

#[test]
fn constant_folding_collapses_lit_plus_lit() {
    // Pre-fold: filter is `price > (100 + 50)`. After fold, it's `price > 150`.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(100.0_f64).plus(lit(50.0_f64))))
        .unwrap()
        .into_plan();
    let folded = ConstantFolding.rewrite(plan).unwrap();
    let LogicalPlan::Filter { predicate, .. } = folded else {
        panic!("expected Filter at top");
    };
    let Expr::BinaryOp { right, .. } = predicate else {
        panic!("expected BinaryOp");
    };
    assert!(matches!(*right, Expr::Literal(ScalarValue::Float64(v)) if (v - 150.0).abs() < 1e-12));
}

#[test]
fn constant_folding_short_circuits_true_and_x() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(lit(true).and(col("price").gt(lit(100.0_f64))))
        .unwrap()
        .into_plan();
    let folded = ConstantFolding.rewrite(plan).unwrap();
    let LogicalPlan::Filter { predicate, .. } = folded else {
        panic!()
    };
    // After folding, the AND collapses to just `price > 100`.
    let Expr::BinaryOp { op, .. } = &predicate else {
        panic!("expected BinaryOp, got {predicate:?}")
    };
    assert_eq!(*op, BinaryOperator::Gt);
}

#[test]
fn constant_folding_evaluates_lit_comparisons_to_bool() {
    // 1 < 2  AND  region = 'west'  ->  true AND (region = 'west')
    //                              ->  region = 'west'
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(lit(1_i32).lt(lit(2_i32)).and(col("region").eq(lit("west"))))
        .unwrap()
        .into_plan();
    let folded = ConstantFolding.rewrite(plan).unwrap();
    let LogicalPlan::Filter { predicate, .. } = folded else {
        panic!()
    };
    let Expr::BinaryOp { op, .. } = &predicate else {
        panic!()
    };
    assert_eq!(*op, BinaryOperator::Eq);
}

// =======================================================================
// DeadProjectionElim
// =======================================================================

#[test]
fn identity_projection_is_eliminated() {
    // Project all five columns explicitly, in order — identity. After the
    // dead-projection rule, the Projection node should be gone.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .project(vec![
            col("order_id"),
            col("category"),
            col("region"),
            col("price"),
            col("quantity"),
        ])
        .unwrap()
        .into_plan();
    let cleaned = DeadProjectionElim.rewrite(plan).unwrap();
    assert!(
        matches!(cleaned, LogicalPlan::Scan { .. }),
        "identity Projection should collapse to Scan, got\n{}",
        cleaned.explain()
    );
}

#[test]
fn non_identity_projection_is_kept() {
    // Reorder cols -> not identity.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .project(vec![col("price"), col("region")])
        .unwrap()
        .into_plan();
    let cleaned = DeadProjectionElim.rewrite(plan).unwrap();
    assert!(matches!(cleaned, LogicalPlan::Projection { .. }));
}

#[test]
fn aliased_projection_is_kept() {
    // Renaming makes the projection non-identity even if columns line up.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .project(vec![
            col("order_id"),
            col("category"),
            col("region"),
            col("price"),
            col("quantity").alias("qty"),
        ])
        .unwrap()
        .into_plan();
    let cleaned = DeadProjectionElim.rewrite(plan).unwrap();
    assert!(matches!(cleaned, LogicalPlan::Projection { .. }));
}

// =======================================================================
// LimitPushdown
// =======================================================================

#[test]
fn limit_directly_above_scan_is_folded_into_scan() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .limit(50)
        .into_plan();
    let pushed = LimitPushdown.rewrite(plan).unwrap();
    // Top-level Limit stays in the plan; Scan also carries limit=50.
    let LogicalPlan::Limit { input, n, .. } = &pushed else {
        panic!("expected Limit at top, got\n{}", pushed.explain());
    };
    assert_eq!(*n, 50);
    let LogicalPlan::Scan { limit, .. } = input.as_ref() else {
        panic!("expected Scan under Limit, got\n{}", input.explain());
    };
    assert_eq!(*limit, Some(50));
}

#[test]
fn limit_does_not_fold_through_filter() {
    // Filter changes row count, so we can't read just N rows from the scan
    // and assume we'll have N matching rows.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("price").gt(lit(500.0_f64)))
        .unwrap()
        .limit(50)
        .into_plan();
    let pushed = LimitPushdown.rewrite(plan).unwrap();
    if let LogicalPlan::Scan { limit, .. } = scan_node(&pushed) {
        assert_eq!(*limit, None, "limit must not push through Filter");
    } else {
        panic!();
    }
}

#[test]
fn limit_does_not_fold_through_aggregate() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .aggregate(vec![col("region")], vec![count_star().alias("n")])
        .unwrap()
        .limit(2)
        .into_plan();
    let pushed = LimitPushdown.rewrite(plan).unwrap();
    if let LogicalPlan::Scan { limit, .. } = scan_node(&pushed) {
        assert_eq!(*limit, None, "limit must not push through Aggregate");
    } else {
        panic!();
    }
}

#[test]
fn limit_folds_through_projection() {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .project(vec![col("order_id"), col("price")])
        .unwrap()
        .limit(25)
        .into_plan();
    let pushed = LimitPushdown.rewrite(plan).unwrap();
    if let LogicalPlan::Scan { limit, .. } = scan_node(&pushed) {
        assert_eq!(*limit, Some(25));
    } else {
        panic!();
    }
}

#[test]
fn nested_limits_take_the_smaller() {
    // limit(20) on top of limit(100): scan should see the tighter cap.
    // Outer limit stays in the plan; we only check the scan's limit.
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .limit(100)
        .limit(20)
        .into_plan();
    let pushed = LimitPushdown.rewrite(plan).unwrap();
    if let LogicalPlan::Scan { limit, .. } = scan_node(&pushed) {
        assert_eq!(*limit, Some(20));
    } else {
        panic!();
    }
}

fn fold_predicate(predicate: Expr) -> Expr {
    let plan = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(predicate)
        .unwrap()
        .into_plan();
    let LogicalPlan::Filter { predicate, .. } = ConstantFolding.rewrite(plan).unwrap() else {
        panic!("expected Filter at top");
    };
    predicate
}

#[test]
fn constant_folding_flips_not_of_a_literal() {
    // NOT (1 > 2) -> NOT false -> true, then true AND x -> x.
    let p = fold_predicate(not(lit(1_i32).gt(lit(2_i32))).and(col("region").eq(lit("west"))));
    assert_eq!(p.display_name(), "(region = \"west\")");
    let p = fold_predicate(not(lit(true)));
    assert!(
        matches!(p, Expr::Literal(ScalarValue::Boolean(false))),
        "{p:?}"
    );
}

#[test]
fn constant_folding_leaves_not_null_unknown() {
    // NOT NULL is NULL, not true or false, so it must not fold to a bool.
    let p = fold_predicate(not(Expr::Literal(ScalarValue::Null)));
    assert!(
        !matches!(p, Expr::Literal(ScalarValue::Boolean(_))),
        "NOT NULL folded to a definite bool: {p:?}"
    );
}

#[test]
fn constant_folding_evaluates_null_checks_on_literals() {
    let p = fold_predicate(Expr::Literal(ScalarValue::Null).is_null());
    assert!(
        matches!(p, Expr::Literal(ScalarValue::Boolean(true))),
        "{p:?}"
    );
    let p = fold_predicate(lit(3_i32).is_null());
    assert!(
        matches!(p, Expr::Literal(ScalarValue::Boolean(false))),
        "{p:?}"
    );
    let p = fold_predicate(lit(3_i32).is_not_null());
    assert!(
        matches!(p, Expr::Literal(ScalarValue::Boolean(true))),
        "{p:?}"
    );
    // Folding reaches inside: (1 + 2) IS NULL -> false.
    let p = fold_predicate(lit(1_i32).plus(lit(2_i32)).is_null());
    assert!(
        matches!(p, Expr::Literal(ScalarValue::Boolean(false))),
        "{p:?}"
    );
    // A column is left alone.
    let p = fold_predicate(col("price").is_null());
    assert_eq!(p.display_name(), "(price IS NULL)");
}

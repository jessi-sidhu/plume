//! Plan-shape tests for predicate pushdown across joins, plus a few
//! ColumnPruning-across-joins cases. The diff harness in plume-bench
//! verifies end-to-end correctness against DuckDB; these tests verify
//! that the optimizer arrives at the *expected* plan shape — in
//! particular, that outer-join safety rules are honoured (predicates
//! that would discard NULL-padded rows must NOT be pushed past the
//! join).

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use plume_core::{col, lit, DataFrame, JoinType, LogicalPlan};
use plume_optimizer::{optimize, pushdown_predicates};

fn customers_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
    ]))
}
fn orders_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("customer_id", DataType::Int64, true),
        Field::new("amount", DataType::Float64, false),
    ]))
}

fn customers() -> DataFrame {
    DataFrame::scan_parquet("customers.parquet", customers_schema())
}
fn orders() -> DataFrame {
    DataFrame::scan_parquet("orders.parquet", orders_schema())
}

// ---- helpers -----------------------------------------------------------

/// Walk the plan and collect Filter predicates as their display strings,
/// in pre-order.
fn filter_predicates(plan: &LogicalPlan) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(p: &LogicalPlan, out: &mut Vec<String>) {
        match p {
            LogicalPlan::Filter { input, predicate, .. } => {
                out.push(predicate.display_name());
                walk(input, out);
            }
            LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Limit { input, .. } => walk(input, out),
            LogicalPlan::Join { left, right, .. } => {
                walk(left, out);
                walk(right, out);
            }
            LogicalPlan::Scan { .. } => {}
        }
    }
    walk(plan, &mut out);
    out
}

// ---- Inner join: per-side and both-side -------------------------------

#[test]
fn inner_join_pushes_left_only_predicate_into_left_subtree() {
    // Filter on customers.name above the join: should land as a Filter
    // wrapping the customers scan.
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Inner).unwrap()
        .filter(col("name").eq(lit("alice"))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();

    // After pushdown there should be NO Filter above the Join — the
    // single conjunct goes entirely to the left side.
    let preds_above_join = filter_predicates_above_join(&optimized);
    assert!(
        preds_above_join.is_empty(),
        "expected no Filter above the Join, got: {preds_above_join:?}"
    );
    // And the left subtree should now have a Filter on `name = "alice"`.
    let left = left_subtree(&optimized);
    assert!(
        contains_filter(left, "(name = \"alice\")"),
        "expected left subtree to have Filter on `name = \"alice\"`; got:\n{}",
        left.explain()
    );
}

#[test]
fn inner_join_pushes_right_only_predicate_into_right_subtree() {
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Inner).unwrap()
        .filter(col("amount").gt(lit(500.0_f64))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();

    assert!(filter_predicates_above_join(&optimized).is_empty());
    let right = right_subtree(&optimized);
    assert!(
        contains_filter(right, "(amount > 500)"),
        "expected right subtree to have Filter on `amount > 500`; got:\n{}",
        right.explain()
    );
}

#[test]
fn inner_join_splits_two_conjuncts_across_sides() {
    // `name = "alice" AND amount > 500` — each conjunct goes to its
    // own side; no Filter remains above the join.
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Inner).unwrap()
        .filter(col("name").eq(lit("alice")).and(col("amount").gt(lit(500.0_f64)))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();

    assert!(filter_predicates_above_join(&optimized).is_empty());
    assert!(contains_filter(left_subtree(&optimized), "(name = \"alice\")"));
    assert!(contains_filter(right_subtree(&optimized), "(amount > 500)"));
}

#[test]
fn inner_join_keeps_both_side_predicate_above() {
    // `customers.id = orders.customer_id` is a both-side conjunct. It
    // should stay above the join (Phase 5 doesn't fold into Join.filter).
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Inner).unwrap()
        .filter(col("region").eq(lit("west")).and(col("id").eq(col("customer_id")))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();

    let preds_above = filter_predicates_above_join(&optimized);
    // The region = "west" left-only conjunct pushed left; the both-side
    // id = customer_id stays above.
    assert!(
        preds_above.iter().any(|p| p.contains("id") && p.contains("customer_id")),
        "expected both-side conjunct to stay above the Join, got: {preds_above:?}"
    );
}

// ---- Left join: right-only predicate must NOT push -------------------

#[test]
fn left_join_does_not_push_right_only_predicate() {
    // The classic counterexample. Filter `amount > 100` above a LEFT
    // JOIN must stay above; pushing it into the right subtree would
    // discard the very right rows whose absence the LEFT join is
    // supposed to preserve as NULL-padded — and the predicate above
    // would then evaluate against a NULL `amount`, not a real one,
    // changing the answer.
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Left).unwrap()
        .filter(col("amount").gt(lit(100.0_f64))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();

    let preds_above = filter_predicates_above_join(&optimized);
    assert!(
        preds_above.iter().any(|p| p.contains("amount")),
        "right-only predicate must stay above a LEFT join, but no Filter \
         on amount was found above the Join. plan:\n{}",
        optimized.explain()
    );
    // And the right subtree must NOT have a Filter on amount.
    let right = right_subtree(&optimized);
    assert!(
        !contains_filter(right, "(amount > 100)"),
        "right-only predicate was incorrectly pushed into the right subtree of \
         a LEFT join. plan:\n{}",
        optimized.explain()
    );
}

#[test]
fn left_join_does_push_left_only_predicate() {
    // Symmetric to the above: a left-only predicate on a LEFT join
    // *can* push, because the left side is the preserved side and
    // shrinking it just produces fewer (still-correctly-padded) rows.
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Left).unwrap()
        .filter(col("name").eq(lit("alice"))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();

    assert!(filter_predicates_above_join(&optimized).is_empty());
    assert!(contains_filter(left_subtree(&optimized), "(name = \"alice\")"));
}

// ---- Right join: left-only mirrors Left ------------------------------

#[test]
fn right_join_does_not_push_left_only_predicate() {
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Right).unwrap()
        .filter(col("name").eq(lit("alice"))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();

    let preds_above = filter_predicates_above_join(&optimized);
    assert!(
        preds_above.iter().any(|p| p.contains("name")),
        "left-only predicate must stay above a RIGHT join. plan:\n{}",
        optimized.explain()
    );
    assert!(!contains_filter(left_subtree(&optimized), "(name = \"alice\")"));
}

#[test]
fn right_join_does_push_right_only_predicate() {
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Right).unwrap()
        .filter(col("amount").gt(lit(500.0_f64))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    assert!(filter_predicates_above_join(&optimized).is_empty());
    assert!(contains_filter(right_subtree(&optimized), "(amount > 500)"));
}

// ---- Full join: nothing pushes ---------------------------------------

#[test]
fn full_join_does_not_push_anything() {
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Full).unwrap()
        .filter(col("name").eq(lit("alice")).and(col("amount").gt(lit(500.0_f64)))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();

    let preds_above = filter_predicates_above_join(&optimized);
    assert!(
        preds_above.iter().any(|p| p.contains("name") && p.contains("amount")),
        "FULL join must keep all predicates above. plan:\n{}",
        optimized.explain()
    );
    assert!(!contains_filter(left_subtree(&optimized), "(name = \"alice\")"));
    assert!(!contains_filter(right_subtree(&optimized), "(amount > 500)"));
}

// ---- Semi/Anti: left-only predicates push -----------------------------

#[test]
fn semi_join_pushes_left_only_predicate() {
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Semi).unwrap()
        .filter(col("name").eq(lit("alice"))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    assert!(filter_predicates_above_join(&optimized).is_empty());
    assert!(contains_filter(left_subtree(&optimized), "(name = \"alice\")"));
}

#[test]
fn anti_join_pushes_left_only_predicate() {
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Anti).unwrap()
        .filter(col("name").eq(lit("alice"))).unwrap()
        .into_plan();
    let optimized = pushdown_predicates(plan).unwrap();
    assert!(filter_predicates_above_join(&optimized).is_empty());
    assert!(contains_filter(left_subtree(&optimized), "(name = \"alice\")"));
}

// ---- Pushdown reaches Parquet scan ------------------------------------

#[test]
fn pushdown_through_join_reaches_scan_pushdown_predicates() {
    // After splitting + recursing, a left-only predicate pushed into
    // the customers scan should also attach as a row-group-pruning
    // hint on that scan (the existing scan-pushdown rule fires on
    // the wrapped Filter -> Scan subtree).
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Inner).unwrap()
        .filter(col("region").eq(lit("west"))).unwrap()
        .into_plan();
    let optimized = optimize(plan).unwrap();

    // Walk to the customers scan and check pushdown_predicates contains
    // (region = "west").
    let customers_scan = find_scan_by_path(&optimized, "customers.parquet")
        .expect("customers scan should still exist");
    if let LogicalPlan::Scan { pushdown_predicates, .. } = customers_scan {
        let names: Vec<String> = pushdown_predicates.iter().map(|p| p.display_name()).collect();
        assert!(
            names.iter().any(|n| n.contains("region") && n.contains("west")),
            "expected (region = \"west\") in customers scan pushdown_predicates, got {names:?}"
        );
    } else {
        unreachable!()
    }
}

// ---- ColumnPruning across joins --------------------------------------

#[test]
fn column_pruning_narrows_each_join_side_independently() {
    // SELECT customers.name, orders.amount FROM customers JOIN orders ON id = customer_id.
    // After pruning, the customers scan should keep only (id, name)
    // and the orders scan should keep only (customer_id, amount).
    let plan = customers()
        .join(orders(), vec![(col("id"), col("customer_id"))], JoinType::Inner).unwrap()
        .project(vec![col("name"), col("amount")]).unwrap()
        .into_plan();
    let optimized = optimize(plan).unwrap();

    let cust = find_scan_by_path(&optimized, "customers.parquet").unwrap();
    let ord = find_scan_by_path(&optimized, "orders.parquet").unwrap();
    if let LogicalPlan::Scan { projection: Some(cols), .. } = cust {
        assert!(cols.contains(&"name".to_string()));
        assert!(cols.contains(&"id".to_string()), "join key must be kept");
        assert!(!cols.contains(&"region".to_string()), "region not referenced");
    } else {
        panic!("customers scan was not column-pruned");
    }
    if let LogicalPlan::Scan { projection: Some(cols), .. } = ord {
        assert!(cols.contains(&"amount".to_string()));
        assert!(cols.contains(&"customer_id".to_string()), "join key must be kept");
        assert!(!cols.contains(&"id".to_string()), "orders.id not referenced");
    } else {
        panic!("orders scan was not column-pruned");
    }
}

// ---- Helper plumbing --------------------------------------------------

fn filter_predicates_above_join(plan: &LogicalPlan) -> Vec<String> {
    // Walk down until we hit a Join, collecting Filter predicates we
    // pass through. Stop at the Join (don't descend).
    let mut out = Vec::new();
    fn walk(p: &LogicalPlan, out: &mut Vec<String>) {
        match p {
            LogicalPlan::Filter { input, predicate, .. } => {
                out.push(predicate.display_name());
                walk(input, out);
            }
            LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Limit { input, .. } => walk(input, out),
            LogicalPlan::Join { .. } => {}
            LogicalPlan::Scan { .. } => {}
        }
    }
    walk(plan, &mut out);
    out
}

fn left_subtree(plan: &LogicalPlan) -> &LogicalPlan {
    fn find_join(p: &LogicalPlan) -> &LogicalPlan {
        match p {
            LogicalPlan::Join { .. } => p,
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Limit { input, .. } => find_join(input),
            LogicalPlan::Scan { .. } => panic!("no Join in plan"),
        }
    }
    let LogicalPlan::Join { left, .. } = find_join(plan) else { unreachable!() };
    left
}

fn right_subtree(plan: &LogicalPlan) -> &LogicalPlan {
    fn find_join(p: &LogicalPlan) -> &LogicalPlan {
        match p {
            LogicalPlan::Join { .. } => p,
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Limit { input, .. } => find_join(input),
            LogicalPlan::Scan { .. } => panic!("no Join in plan"),
        }
    }
    let LogicalPlan::Join { right, .. } = find_join(plan) else { unreachable!() };
    right
}

fn contains_filter(plan: &LogicalPlan, predicate_text: &str) -> bool {
    match plan {
        LogicalPlan::Filter { input, predicate, .. } => {
            if predicate.display_name() == predicate_text {
                true
            } else {
                contains_filter(input, predicate_text)
            }
        }
        LogicalPlan::Projection { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Limit { input, .. } => contains_filter(input, predicate_text),
        LogicalPlan::Join { left, right, .. } => {
            contains_filter(left, predicate_text) || contains_filter(right, predicate_text)
        }
        LogicalPlan::Scan { .. } => false,
    }
}

fn find_scan_by_path<'a>(plan: &'a LogicalPlan, suffix: &str) -> Option<&'a LogicalPlan> {
    match plan {
        LogicalPlan::Scan { source, .. } => {
            if source.display_path().ends_with(suffix) {
                Some(plan)
            } else {
                None
            }
        }
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Projection { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Limit { input, .. } => find_scan_by_path(input, suffix),
        LogicalPlan::Join { left, right, .. } => {
            find_scan_by_path(left, suffix).or_else(|| find_scan_by_path(right, suffix))
        }
    }
}

#[test]
fn filter_predicates_helper_smoke() {
    // Smoke-check the helper itself doesn't crash on a join-free plan.
    let plan = customers()
        .filter(col("region").eq(lit("west"))).unwrap()
        .into_plan();
    let preds = filter_predicates(&plan);
    assert_eq!(preds, vec!["(region = \"west\")"]);
}

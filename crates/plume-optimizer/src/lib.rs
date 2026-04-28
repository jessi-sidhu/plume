//! plume-optimizer: logical-plan rewrites.
//!
//! Week 8: column pruning. Walks the plan top-down collecting the column
//! names that need to flow up out of each node, and rewrites Scan nodes to
//! read only those columns. The Scan's `schema` is restricted to the kept
//! columns so downstream operators see the right field set.
//!
//! This is intentionally a single-pass function rather than a full
//! `OptimizerRule` trait + driver. The trait + fixed-point driver lands in
//! Week 10 alongside the second and third rules (predicate pushdown,
//! constant folding) when there's enough surface to motivate the framework.

use std::collections::HashSet;

use plume_core::{BinaryOperator, Expr, LogicalPlan, Result, ScalarValue, ScanSource};

/// Apply column pruning. Safe to call on any plan; no-ops if there's
/// nothing to prune.
pub fn prune_columns(plan: LogicalPlan) -> Result<LogicalPlan> {
    // Top-level required = every column the plan currently emits. The
    // recursive pass refines this as it descends.
    let required: HashSet<String> = plan
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    prune_with_required(plan, &required)
}

fn prune_with_required(
    plan: LogicalPlan,
    required_from_above: &HashSet<String>,
) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Scan { source, schema, projection, pushdown_predicates } => {
            // CSV is row-oriented — column projection only saves parse CPU,
            // never I/O — so skip it. The reader still hands back every
            // column; downstream operators trim what they don't need.
            if matches!(source, ScanSource::Csv { .. }) {
                return Ok(LogicalPlan::Scan { source, schema, projection, pushdown_predicates });
            }
            // If the optimizer has already projected this scan, leave it
            // alone — the user (or a previous pass) made an explicit choice.
            if projection.is_some() {
                return Ok(LogicalPlan::Scan { source, schema, projection, pushdown_predicates });
            }
            // Keep file-schema column order so consumers reading positionally
            // (parquet metadata, csv header) see what they expect.
            let kept: Vec<String> = schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| required_from_above.contains(n))
                .collect();
            // If all columns are required, projection is a no-op — leave None.
            if kept.len() == schema.fields().len() {
                return Ok(LogicalPlan::Scan { source, schema, projection: None, pushdown_predicates });
            }
            // If nothing is required (e.g. SELECT 1 FROM t), keep at least
            // one column so the row count is preserved. Pick the first.
            let kept = if kept.is_empty() {
                vec![schema.field(0).name().clone()]
            } else {
                kept
            };
            let scan = LogicalPlan::scan_with_projection(source, schema, kept)?;
            LogicalPlan::with_pushdown_predicates(scan, pushdown_predicates)
        }

        LogicalPlan::Filter { input, predicate, schema: _ } => {
            // Filter needs its predicate's columns plus whatever flows up.
            let mut needed = required_from_above.clone();
            collect_columns(&predicate, &mut needed);
            let pruned_input = prune_with_required(*input, &needed)?;
            LogicalPlan::filter(pruned_input, predicate)
        }

        LogicalPlan::Projection { input, exprs, schema: _ } => {
            // The projection re-shapes its input into `exprs`. The input
            // need only supply columns referenced by those expressions —
            // `required_from_above` is irrelevant here (the projection
            // doesn't pass through extra columns regardless).
            let needed = exprs_columns(&exprs);
            let pruned_input = prune_with_required(*input, &needed)?;
            LogicalPlan::project(pruned_input, exprs)
        }

        LogicalPlan::Aggregate { input, group_exprs, agg_exprs, schema: _ } => {
            // Aggregate's input needs every column referenced by either
            // the grouping or the aggregate-arg expressions.
            let mut needed = exprs_columns(&group_exprs);
            for e in &agg_exprs {
                collect_columns(e, &mut needed);
            }
            let pruned_input = prune_with_required(*input, &needed)?;
            LogicalPlan::aggregate(pruned_input, group_exprs, agg_exprs)
        }

        LogicalPlan::Limit { input, n, schema: _ } => {
            let pruned_input = prune_with_required(*input, required_from_above)?;
            Ok(LogicalPlan::limit(pruned_input, n))
        }
    }
}

fn exprs_columns(exprs: &[Expr]) -> HashSet<String> {
    let mut out = HashSet::new();
    for e in exprs {
        collect_columns(e, &mut out);
    }
    out
}

/// Walk an `Expr` collecting every `Column` reference into `out`.
fn collect_columns(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Column(name) => {
            out.insert(name.clone());
        }
        Expr::Literal(_) => {}
        Expr::BinaryOp { left, right, .. } => {
            collect_columns(left, out);
            collect_columns(right, out);
        }
        Expr::Aggregate { arg, .. } => collect_columns(arg, out),
        Expr::Alias { expr, .. } => collect_columns(expr, out),
    }
}

// =======================================================================
// Predicate pushdown — Week 9
// =======================================================================

/// Walks the plan, finds `Filter` nodes sitting directly on a Parquet
/// `Scan` (or on a Scan with intermediate Projection/Limit), splits the
/// filter's predicate into its AND-clauses, and pushes every clause that
/// qualifies as a "scan-pushable comparison" into the Scan's
/// `pushdown_predicates`. The Filter itself is *not* removed — pushdown
/// is a row-group-skipping hint, not a replacement for filtering.
///
/// Pushable form: `col OP literal` (or `literal OP col`, normalized) where
/// `OP` is one of `=, <>, <, <=, >, >=`. Conjunctions decompose to their
/// clauses. Anything else (OR, NOT, function calls, expressions, NULL
/// literals) is left in the Filter and does no work at the scan layer.
pub fn pushdown_predicates(plan: LogicalPlan) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Filter { input, predicate, schema: _ } => {
            // First recurse so any nested Filters get processed.
            let pushed_input = pushdown_predicates(*input)?;
            let pushable = collect_pushable(&predicate);
            let new_input = match (pushable.is_empty(), &pushed_input) {
                (true, _) => pushed_input,
                (false, LogicalPlan::Scan { source, .. })
                    if matches!(source, ScanSource::Parquet { .. }) =>
                {
                    LogicalPlan::with_pushdown_predicates(pushed_input, pushable)?
                }
                _ => pushed_input,
            };
            LogicalPlan::filter(new_input, predicate)
        }
        LogicalPlan::Scan { .. } => Ok(plan),
        LogicalPlan::Projection { input, exprs, schema: _ } => {
            let new_input = pushdown_predicates(*input)?;
            LogicalPlan::project(new_input, exprs)
        }
        LogicalPlan::Aggregate { input, group_exprs, agg_exprs, schema: _ } => {
            let new_input = pushdown_predicates(*input)?;
            LogicalPlan::aggregate(new_input, group_exprs, agg_exprs)
        }
        LogicalPlan::Limit { input, n, schema: _ } => {
            let new_input = pushdown_predicates(*input)?;
            Ok(LogicalPlan::limit(new_input, n))
        }
    }
}

/// Convenience: run column pruning *after* predicate pushdown. Pushdown
/// goes first so the column-pruning pass sees the predicate columns
/// already attached to the Scan (and so they're counted as required).
pub fn optimize(plan: LogicalPlan) -> Result<LogicalPlan> {
    let p = pushdown_predicates(plan)?;
    prune_columns(p)
}

/// Decompose `e` into its AND-clauses and return the subset that is a
/// scan-pushable `col OP literal` comparison (operands swapped where
/// needed so the column is on the left).
fn collect_pushable(e: &Expr) -> Vec<Expr> {
    let mut out = Vec::new();
    walk_conjunction(e, &mut out);
    out
}

fn walk_conjunction(e: &Expr, out: &mut Vec<Expr>) {
    match e {
        Expr::BinaryOp { left, op: BinaryOperator::And, right } => {
            walk_conjunction(left, out);
            walk_conjunction(right, out);
        }
        // Strip aliases — they're cosmetic.
        Expr::Alias { expr, .. } => walk_conjunction(expr, out),
        other => {
            if let Some(normalized) = normalize_pushable(other) {
                out.push(normalized);
            }
        }
    }
}

/// Recognise `col OP lit` (or `lit OP col`). Returns the predicate with
/// the column on the left, op flipped if we swapped sides. NULL literals
/// produce no pushdown — three-valued logic at the row-group level is
/// noise we'd have to handle and the per-row Filter still does it
/// correctly.
fn normalize_pushable(e: &Expr) -> Option<Expr> {
    let Expr::BinaryOp { left, op, right } = e else { return None };
    if !op.is_comparison() {
        return None;
    }
    let l = strip_alias(left);
    let r = strip_alias(right);
    let (col_side, lit_side, op_for_col_left) = match (l, r) {
        (Expr::Column(_), Expr::Literal(v)) if !is_null(v) => {
            (l.clone(), Expr::Literal(v.clone()), *op)
        }
        (Expr::Literal(v), Expr::Column(_)) if !is_null(v) => {
            (r.clone(), Expr::Literal(v.clone()), flip(*op))
        }
        _ => return None,
    };
    Some(Expr::BinaryOp {
        left: Box::new(col_side),
        op: op_for_col_left,
        right: Box::new(lit_side),
    })
}

fn strip_alias(e: &Expr) -> &Expr {
    match e {
        Expr::Alias { expr, .. } => strip_alias(expr),
        other => other,
    }
}

fn is_null(v: &ScalarValue) -> bool {
    matches!(v, ScalarValue::Null)
}

fn flip(op: BinaryOperator) -> BinaryOperator {
    use BinaryOperator::*;
    match op {
        Eq => Eq,
        NotEq => NotEq,
        Lt => Gt,
        LtEq => GtEq,
        Gt => Lt,
        GtEq => LtEq,
        // Logical/arithmetic shouldn't reach here (filtered by is_comparison).
        And | Or | Plus | Minus | Multiply | Divide => op,
    }
}


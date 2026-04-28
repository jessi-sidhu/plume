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

use plume_core::{Expr, LogicalPlan, Result, ScanSource};

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
        LogicalPlan::Scan { source, schema, projection } => {
            // CSV is row-oriented — column projection only saves parse CPU,
            // never I/O — so skip it. The reader still hands back every
            // column; downstream operators trim what they don't need.
            if matches!(source, ScanSource::Csv { .. }) {
                return Ok(LogicalPlan::Scan { source, schema, projection });
            }
            // If the optimizer has already projected this scan, leave it
            // alone — the user (or a previous pass) made an explicit choice.
            if projection.is_some() {
                return Ok(LogicalPlan::Scan { source, schema, projection });
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
                return Ok(LogicalPlan::Scan { source, schema, projection: None });
            }
            // If nothing is required (e.g. SELECT 1 FROM t), keep at least
            // one column so the row count is preserved. Pick the first.
            let kept = if kept.is_empty() {
                vec![schema.field(0).name().clone()]
            } else {
                kept
            };
            LogicalPlan::scan_with_projection(source, schema, kept)
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


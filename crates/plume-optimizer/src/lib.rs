//! plume-optimizer: logical-plan rewrites driven by an `OptimizerRule` trait
//! and a fixed-point driver.
//!
//! Rule list, in default order:
//!   1. ConstantFolding         — fold literal-only sub-expressions
//!   2. DeadProjectionElim      — collapse identity / nested projections
//!   3. PredicatePushdown       — push col-OP-lit predicates into Parquet scans
//!   4. ColumnPruning           — restrict scans to required columns
//!   5. LimitPushdown           — fold a top-level Limit into the Scan
//!
//! Order matters. Pushdown writes predicate columns into the Scan's
//! pushdown list; column pruning then has to keep those columns. Limit
//! pushdown runs last so it sees the projected scan.
//!
//! `Optimizer::default().optimize(plan)` runs the whole pipeline in a
//! fixed point. `prune_columns` / `pushdown_predicates` remain as
//! single-rule entry points for tests that want to see one pass in
//! isolation.

use std::collections::HashSet;

use plume_core::{BinaryOperator, Expr, LogicalPlan, Result, ScalarValue, ScanSource};

// =======================================================================
// Trait + driver
// =======================================================================

pub trait OptimizerRule: Send + Sync {
    fn name(&self) -> &str;
    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan>;
}

pub struct Optimizer {
    rules: Vec<Box<dyn OptimizerRule>>,
    /// Hard ceiling on rewrite passes so a misbehaving rule that flip-flops
    /// the plan doesn't loop forever.
    max_passes: usize,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self {
            rules: vec![
                Box::new(ConstantFolding),
                Box::new(DeadProjectionElim),
                Box::new(PredicatePushdown),
                Box::new(ColumnPruning),
                Box::new(LimitPushdown),
            ],
            max_passes: 8,
        }
    }
}

impl Optimizer {
    pub fn with_rules(rules: Vec<Box<dyn OptimizerRule>>) -> Self {
        Self { rules, max_passes: 8 }
    }

    /// Apply every rule in order; repeat until a full pass changes nothing
    /// (or until `max_passes` is exhausted).
    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let mut current = plan;
        for _ in 0..self.max_passes {
            let mut changed = false;
            for rule in &self.rules {
                let before = format!("{}", current); // cheap text-form fingerprint
                let after = rule.rewrite(current.clone())?;
                let after_text = format!("{}", after);
                if after_text != before {
                    changed = true;
                }
                current = after;
            }
            if !changed {
                break;
            }
        }
        Ok(current)
    }
}

/// Backwards-compatible single-rule entry points (used by tests that want
/// to inspect one pass at a time).
pub fn prune_columns(plan: LogicalPlan) -> Result<LogicalPlan> {
    ColumnPruning.rewrite(plan)
}
pub fn pushdown_predicates(plan: LogicalPlan) -> Result<LogicalPlan> {
    PredicatePushdown.rewrite(plan)
}
pub fn optimize(plan: LogicalPlan) -> Result<LogicalPlan> {
    Optimizer::default().optimize(plan)
}

// =======================================================================
// Rule: ColumnPruning  (Week 8 logic, behind the trait)
// =======================================================================

pub struct ColumnPruning;

impl OptimizerRule for ColumnPruning {
    fn name(&self) -> &str { "column_pruning" }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let required: HashSet<String> = plan
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        prune_with_required(plan, &required)
    }
}

fn prune_with_required(
    plan: LogicalPlan,
    required_from_above: &HashSet<String>,
) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Scan { source, schema, projection, pushdown_predicates, limit } => {
            if matches!(source, ScanSource::Csv { .. }) {
                return Ok(LogicalPlan::Scan { source, schema, projection, pushdown_predicates, limit });
            }
            if projection.is_some() {
                return Ok(LogicalPlan::Scan { source, schema, projection, pushdown_predicates, limit });
            }
            // Predicate columns must be retained even though the predicate
            // already lives in pushdown_predicates — the per-row Filter
            // (still in the plan above) needs them.
            let mut needed = required_from_above.clone();
            for pred in &pushdown_predicates {
                collect_columns(pred, &mut needed);
            }
            let kept: Vec<String> = schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| needed.contains(n))
                .collect();
            if kept.len() == schema.fields().len() {
                return Ok(LogicalPlan::Scan {
                    source, schema, projection: None, pushdown_predicates, limit,
                });
            }
            let kept = if kept.is_empty() {
                vec![schema.field(0).name().clone()]
            } else {
                kept
            };
            let scan = LogicalPlan::scan_with_projection(source, schema, kept)?;
            let scan = LogicalPlan::with_pushdown_predicates(scan, pushdown_predicates)?;
            LogicalPlan::with_scan_limit(scan, limit)
        }
        LogicalPlan::Filter { input, predicate, schema: _ } => {
            let mut needed = required_from_above.clone();
            collect_columns(&predicate, &mut needed);
            let pruned_input = prune_with_required(*input, &needed)?;
            LogicalPlan::filter(pruned_input, predicate)
        }
        LogicalPlan::Projection { input, exprs, schema: _ } => {
            let needed = exprs_columns(&exprs);
            let pruned_input = prune_with_required(*input, &needed)?;
            LogicalPlan::project(pruned_input, exprs)
        }
        LogicalPlan::Aggregate { input, group_exprs, agg_exprs, schema: _ } => {
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
        LogicalPlan::Join { left, right, on, filter, join_type, .. } => {
            // Phase 1 passthrough: do NOT prune across joins yet. Recurse
            // into each side with that side's full schema as required, so
            // the children come back schema-equal and the rebuilt Join
            // has the same output schema. Cross-join pruning lands in
            // the join-aware optimizer pass.
            let left_required: HashSet<String> = left
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            let right_required: HashSet<String> = right
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            let new_left = prune_with_required(*left, &left_required)?;
            let new_right = prune_with_required(*right, &right_required)?;
            LogicalPlan::join(new_left, new_right, on, filter, join_type)
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

fn collect_columns(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Column(name) => { out.insert(name.clone()); }
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
// Rule: PredicatePushdown  (Week 9 logic, behind the trait)
// =======================================================================

pub struct PredicatePushdown;

impl OptimizerRule for PredicatePushdown {
    fn name(&self) -> &str { "predicate_pushdown" }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        push_through(plan)
    }
}

fn push_through(plan: LogicalPlan) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Filter { input, predicate, schema: _ } => {
            let pushed_input = push_through(*input)?;
            let pushable = collect_pushable(&predicate);
            let new_input = match (pushable.is_empty(), &pushed_input) {
                (true, _) => pushed_input,
                (false, LogicalPlan::Scan { source, pushdown_predicates: existing, .. })
                    if matches!(source, ScanSource::Parquet { .. }) =>
                {
                    // De-dupe by string form so re-running the rule is idempotent.
                    let mut all = existing.clone();
                    let existing_text: HashSet<String> =
                        existing.iter().map(|p| p.display_name()).collect();
                    for p in pushable {
                        if !existing_text.contains(&p.display_name()) {
                            all.push(p);
                        }
                    }
                    LogicalPlan::with_pushdown_predicates(pushed_input, all)?
                }
                _ => pushed_input,
            };
            LogicalPlan::filter(new_input, predicate)
        }
        LogicalPlan::Scan { .. } => Ok(plan),
        LogicalPlan::Projection { input, exprs, schema: _ } => {
            let new_input = push_through(*input)?;
            LogicalPlan::project(new_input, exprs)
        }
        LogicalPlan::Aggregate { input, group_exprs, agg_exprs, schema: _ } => {
            let new_input = push_through(*input)?;
            LogicalPlan::aggregate(new_input, group_exprs, agg_exprs)
        }
        LogicalPlan::Limit { input, n, schema: _ } => {
            let new_input = push_through(*input)?;
            Ok(LogicalPlan::limit(new_input, n))
        }
        LogicalPlan::Join { left, right, on, filter, join_type, .. } => {
            // Phase 1 passthrough: recurse into both children. Pushdown
            // *across* joins (splitting AND-conjuncts above the join into
            // left/right buckets, with outer-join nullable-side safety)
            // is the load-bearing piece of Phase 5.
            let new_left = push_through(*left)?;
            let new_right = push_through(*right)?;
            LogicalPlan::join(new_left, new_right, on, filter, join_type)
        }
    }
}

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
        Expr::Alias { expr, .. } => walk_conjunction(expr, out),
        other => {
            if let Some(normalized) = normalize_pushable(other) {
                out.push(normalized);
            }
        }
    }
}

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

fn is_null(v: &ScalarValue) -> bool { matches!(v, ScalarValue::Null) }

fn flip(op: BinaryOperator) -> BinaryOperator {
    use BinaryOperator::*;
    match op {
        Eq => Eq,
        NotEq => NotEq,
        Lt => Gt,
        LtEq => GtEq,
        Gt => Lt,
        GtEq => LtEq,
        And | Or | Plus | Minus | Multiply | Divide => op,
    }
}

// =======================================================================
// Rule: ConstantFolding
//
// Folds expression sub-trees whose value is determined at plan time.
// Handles:
//   - lit OP lit  -> lit  (for comparisons + arithmetic on matching types)
//   - true AND x  -> x;   false AND x -> false
//   - false OR x  -> x;   true OR x -> true
// Doesn't try to be clever about NULL — three-valued logic at the plan
// layer is more pain than it's worth for the wins.
// =======================================================================

pub struct ConstantFolding;

impl OptimizerRule for ConstantFolding {
    fn name(&self) -> &str { "constant_folding" }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        rewrite_exprs_in_plan(plan, &fold_expr)
    }
}

fn fold_expr(e: Expr) -> Expr {
    match e {
        Expr::BinaryOp { left, op, right } => {
            let l = fold_expr(*left);
            let r = fold_expr(*right);
            // Boolean short-circuits — only safe when the literal isn't NULL.
            if op == BinaryOperator::And {
                if let Expr::Literal(ScalarValue::Boolean(b)) = &l {
                    return if *b { r } else { Expr::Literal(ScalarValue::Boolean(false)) };
                }
                if let Expr::Literal(ScalarValue::Boolean(b)) = &r {
                    return if *b { l } else { Expr::Literal(ScalarValue::Boolean(false)) };
                }
            }
            if op == BinaryOperator::Or {
                if let Expr::Literal(ScalarValue::Boolean(b)) = &l {
                    return if *b { Expr::Literal(ScalarValue::Boolean(true)) } else { r };
                }
                if let Expr::Literal(ScalarValue::Boolean(b)) = &r {
                    return if *b { Expr::Literal(ScalarValue::Boolean(true)) } else { l };
                }
            }
            // lit OP lit -> lit  (comparisons + arithmetic on matching types)
            if let (Expr::Literal(lv), Expr::Literal(rv)) = (&l, &r) {
                if let Some(folded) = fold_literal_binop(lv, op, rv) {
                    return Expr::Literal(folded);
                }
            }
            Expr::BinaryOp { left: Box::new(l), op, right: Box::new(r) }
        }
        Expr::Aggregate { fun, arg } => Expr::Aggregate { fun, arg: Box::new(fold_expr(*arg)) },
        Expr::Alias { expr, name } => Expr::Alias { expr: Box::new(fold_expr(*expr)), name },
        leaf => leaf,
    }
}

fn fold_literal_binop(l: &ScalarValue, op: BinaryOperator, r: &ScalarValue) -> Option<ScalarValue> {
    use BinaryOperator::*;
    use ScalarValue::*;
    match (op, l, r) {
        // Integer arithmetic — same type only (matches the schema layer's strict rule).
        (Plus, Int32(a), Int32(b)) => Some(Int32(a.wrapping_add(*b))),
        (Minus, Int32(a), Int32(b)) => Some(Int32(a.wrapping_sub(*b))),
        (Multiply, Int32(a), Int32(b)) => Some(Int32(a.wrapping_mul(*b))),
        (Plus, Int64(a), Int64(b)) => Some(Int64(a.wrapping_add(*b))),
        (Minus, Int64(a), Int64(b)) => Some(Int64(a.wrapping_sub(*b))),
        (Multiply, Int64(a), Int64(b)) => Some(Int64(a.wrapping_mul(*b))),
        (Plus, Float64(a), Float64(b)) => Some(Float64(a + b)),
        (Minus, Float64(a), Float64(b)) => Some(Float64(a - b)),
        (Multiply, Float64(a), Float64(b)) => Some(Float64(a * b)),
        // Comparisons across the supported scalar types.
        (Eq, a, b) if a == b => Some(Boolean(true)),
        (Eq, _, _) => Some(Boolean(false)),
        (NotEq, a, b) if a == b => Some(Boolean(false)),
        (NotEq, _, _) => Some(Boolean(true)),
        (Lt, Int32(a), Int32(b)) => Some(Boolean(a < b)),
        (Lt, Int64(a), Int64(b)) => Some(Boolean(a < b)),
        (Lt, Float64(a), Float64(b)) => Some(Boolean(a < b)),
        (Gt, Int32(a), Int32(b)) => Some(Boolean(a > b)),
        (Gt, Int64(a), Int64(b)) => Some(Boolean(a > b)),
        (Gt, Float64(a), Float64(b)) => Some(Boolean(a > b)),
        (LtEq, Int32(a), Int32(b)) => Some(Boolean(a <= b)),
        (LtEq, Int64(a), Int64(b)) => Some(Boolean(a <= b)),
        (LtEq, Float64(a), Float64(b)) => Some(Boolean(a <= b)),
        (GtEq, Int32(a), Int32(b)) => Some(Boolean(a >= b)),
        (GtEq, Int64(a), Int64(b)) => Some(Boolean(a >= b)),
        (GtEq, Float64(a), Float64(b)) => Some(Boolean(a >= b)),
        _ => None,
    }
}

/// Apply `f` to every `Expr` inside `plan` (predicates, projection exprs,
/// aggregate exprs, group exprs). Plan tree shape is unchanged.
fn rewrite_exprs_in_plan(plan: LogicalPlan, f: &impl Fn(Expr) -> Expr) -> Result<LogicalPlan> {
    Ok(match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter { input, predicate, schema: _ } => {
            let new_input = rewrite_exprs_in_plan(*input, f)?;
            LogicalPlan::filter(new_input, f(predicate))?
        }
        LogicalPlan::Projection { input, exprs, schema: _ } => {
            let new_input = rewrite_exprs_in_plan(*input, f)?;
            let new_exprs = exprs.into_iter().map(f).collect();
            LogicalPlan::project(new_input, new_exprs)?
        }
        LogicalPlan::Aggregate { input, group_exprs, agg_exprs, schema: _ } => {
            let new_input = rewrite_exprs_in_plan(*input, f)?;
            let new_group = group_exprs.into_iter().map(f).collect();
            let new_agg = agg_exprs.into_iter().map(f).collect();
            LogicalPlan::aggregate(new_input, new_group, new_agg)?
        }
        LogicalPlan::Limit { input, n, schema: _ } => {
            let new_input = rewrite_exprs_in_plan(*input, f)?;
            LogicalPlan::limit(new_input, n)
        }
        LogicalPlan::Join { left, right, on, filter, join_type, .. } => {
            let new_left = rewrite_exprs_in_plan(*left, f)?;
            let new_right = rewrite_exprs_in_plan(*right, f)?;
            // Apply f to both halves of every key pair, plus the residual
            // filter. f is schema-preserving (it folds expression sub-trees
            // to literals), so the rebuilt Join has the same output schema.
            let new_on: Vec<(Expr, Expr)> = on
                .into_iter()
                .map(|(l, r)| (f(l), f(r)))
                .collect();
            let new_filter = filter.map(f);
            LogicalPlan::join(new_left, new_right, new_on, new_filter, join_type)?
        }
    })
}

// =======================================================================
// Rule: DeadProjectionElim
//
// Drops a Projection that is the identity over its input — i.e. every
// expr is a Column reference to the corresponding input field, in the
// same order, with no renaming. This shows up after column pruning
// shrinks an upstream scan to exactly what the projection wanted.
// =======================================================================

pub struct DeadProjectionElim;

impl OptimizerRule for DeadProjectionElim {
    fn name(&self) -> &str { "dead_projection_elim" }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Projection { input, exprs, schema: _ } => {
                let inner = self.rewrite(*input)?;
                if is_identity_projection(&exprs, &inner) {
                    Ok(inner)
                } else {
                    LogicalPlan::project(inner, exprs)
                }
            }
            LogicalPlan::Filter { input, predicate, schema: _ } => {
                LogicalPlan::filter(self.rewrite(*input)?, predicate)
            }
            LogicalPlan::Aggregate { input, group_exprs, agg_exprs, schema: _ } => {
                LogicalPlan::aggregate(self.rewrite(*input)?, group_exprs, agg_exprs)
            }
            LogicalPlan::Limit { input, n, schema: _ } => {
                Ok(LogicalPlan::limit(self.rewrite(*input)?, n))
            }
            LogicalPlan::Scan { .. } => Ok(plan),
            LogicalPlan::Join { left, right, on, filter, join_type, .. } => {
                let new_left = self.rewrite(*left)?;
                let new_right = self.rewrite(*right)?;
                LogicalPlan::join(new_left, new_right, on, filter, join_type)
            }
        }
    }
}

fn is_identity_projection(exprs: &[Expr], input: &LogicalPlan) -> bool {
    let in_schema = input.schema();
    if exprs.len() != in_schema.fields().len() {
        return false;
    }
    for (i, e) in exprs.iter().enumerate() {
        match e {
            Expr::Column(name) if name == in_schema.field(i).name() => {}
            _ => return false,
        }
    }
    true
}

// =======================================================================
// Rule: LimitPushdown
//
// When a Limit sits directly above a Scan (or above a chain that doesn't
// change row count — currently just Projection), fold the limit into the
// Scan's `limit` field so the executor stops reading early.
//
// Filter and Aggregate change row count, so the Limit must stay above
// them in the plan; we only fold through "row-preserving" nodes.
// =======================================================================

pub struct LimitPushdown;

impl OptimizerRule for LimitPushdown {
    fn name(&self) -> &str { "limit_pushdown" }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Limit { input, n, schema: _ } => {
                let inner = self.rewrite(*input)?;
                // The Limit operator stays in the plan to guarantee the
                // contract — folding the limit into the Scan is just a
                // *hint* (Parquet honors it; CSV ignores it). Keeping the
                // Limit above means CSV still respects the cap.
                let folded_inner = match try_fold_limit(inner, n)? {
                    FoldResult::Folded(p) | FoldResult::NotFolded(p) => p,
                };
                Ok(LogicalPlan::limit(folded_inner, n))
            }
            LogicalPlan::Filter { input, predicate, schema: _ } => {
                LogicalPlan::filter(self.rewrite(*input)?, predicate)
            }
            LogicalPlan::Projection { input, exprs, schema: _ } => {
                LogicalPlan::project(self.rewrite(*input)?, exprs)
            }
            LogicalPlan::Aggregate { input, group_exprs, agg_exprs, schema: _ } => {
                LogicalPlan::aggregate(self.rewrite(*input)?, group_exprs, agg_exprs)
            }
            LogicalPlan::Scan { .. } => Ok(plan),
            LogicalPlan::Join { left, right, on, filter, join_type, .. } => {
                // Joins change row count, so a Limit above never folds
                // through. Recurse into both children to allow
                // Limit-pushdown opportunities inside each subtree.
                let new_left = self.rewrite(*left)?;
                let new_right = self.rewrite(*right)?;
                LogicalPlan::join(new_left, new_right, on, filter, join_type)
            }
        }
    }
}

enum FoldResult {
    Folded(LogicalPlan),
    NotFolded(LogicalPlan),
}

fn try_fold_limit(plan: LogicalPlan, n: usize) -> Result<FoldResult> {
    match plan {
        LogicalPlan::Scan { source, schema, projection, pushdown_predicates, limit } => {
            // Honour the smaller of any pre-existing scan limit and the new one.
            let merged = match limit {
                None => Some(n),
                Some(existing) => Some(existing.min(n)),
            };
            let scan = LogicalPlan::Scan {
                source, schema, projection, pushdown_predicates, limit: merged,
            };
            Ok(FoldResult::Folded(scan))
        }
        // Projection preserves row count (one row in, one row out), so we
        // can fold past it: rebuild Projection over a folded scan.
        LogicalPlan::Projection { input, exprs, schema: _ } => {
            match try_fold_limit(*input, n)? {
                FoldResult::Folded(new_inner) => {
                    Ok(FoldResult::Folded(LogicalPlan::project(new_inner, exprs)?))
                }
                FoldResult::NotFolded(inner) => {
                    Ok(FoldResult::NotFolded(LogicalPlan::project(inner, exprs)?))
                }
            }
        }
        // Limit-on-Limit: take the tighter cap and fold through.
        LogicalPlan::Limit { input, n: inner_n, schema: _ } => {
            let tighter = n.min(inner_n);
            match try_fold_limit(*input, tighter)? {
                FoldResult::Folded(new_inner) => {
                    Ok(FoldResult::Folded(LogicalPlan::limit(new_inner, inner_n)))
                }
                FoldResult::NotFolded(inner) => {
                    Ok(FoldResult::NotFolded(LogicalPlan::limit(inner, inner_n)))
                }
            }
        }
        // Filter and Aggregate change row count — leave the Limit above them.
        other => Ok(FoldResult::NotFolded(other)),
    }
}

// =======================================================================
// explain_optimized — a small ergonomic helper for tests / docs
// =======================================================================

/// Render before/after plans side by side.
pub fn explain_optimized(before: &LogicalPlan, after: &LogicalPlan) -> String {
    format!(
        "BEFORE:\n{}\nAFTER:\n{}",
        before.explain(),
        after.explain()
    )
}

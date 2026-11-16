//! plume-optimizer: logical-plan rewrites driven by an `OptimizerRule` trait
//! and a fixed-point driver.
//!
//! Rule list, in default order:
//!   1. ConstantFolding         — fold literal-only sub-expressions
//!   2. DeadProjectionElim      — collapse identity / nested projections
//!   3. PredicatePushdown       — push col-OP-lit predicates into Parquet scans
//!   4. ColumnPruning           — restrict scans to required columns
//!   5. LimitPushdown           — fold a top-level Limit into the Scan
//!   6. LimitPruning            — tell a filtered scan how many rows a Limit needs
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
                Box::new(LimitPruning),
            ],
            max_passes: 8,
        }
    }
}

impl Optimizer {
    pub fn with_rules(rules: Vec<Box<dyn OptimizerRule>>) -> Self {
        Self {
            rules,
            max_passes: 8,
        }
    }

    /// Apply every rule in order; repeat until a full pass changes nothing
    /// (or until `max_passes` is exhausted).
    ///
    /// Convergence is detected by `PartialEq` on the plan tree. The earlier
    /// implementation rendered both sides to text and string-compared them;
    /// once joins enter the picture, plans get bigger and the per-rule
    /// `format!()` call grows linearly with plan size. Value comparison is
    /// O(plan size) too but avoids the allocation, and "Display unchanged"
    /// is not strictly the same invariant as "plan unchanged" — switching
    /// to PartialEq is also more correct.
    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let mut current = plan;
        for _ in 0..self.max_passes {
            let mut changed = false;
            for rule in &self.rules {
                let before = current.clone();
                let after = rule.rewrite(current)?;
                if after != before {
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
    fn name(&self) -> &str {
        "column_pruning"
    }

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
        LogicalPlan::Scan {
            source,
            schema,
            projection,
            pushdown_predicates,
            limit,
            limit_hint,
        } => {
            if matches!(source, ScanSource::Csv { .. }) {
                return Ok(LogicalPlan::Scan {
                    source,
                    schema,
                    projection,
                    pushdown_predicates,
                    limit,
                    limit_hint,
                });
            }
            if projection.is_some() {
                return Ok(LogicalPlan::Scan {
                    source,
                    schema,
                    projection,
                    pushdown_predicates,
                    limit,
                    limit_hint,
                });
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
                    source,
                    schema,
                    projection: None,
                    pushdown_predicates,
                    limit,
                    limit_hint,
                });
            }
            let kept = if kept.is_empty() {
                vec![schema.field(0).name().clone()]
            } else {
                kept
            };
            let scan = LogicalPlan::scan_with_projection(source, schema, kept)?;
            let scan = LogicalPlan::with_pushdown_predicates(scan, pushdown_predicates)?;
            let scan = LogicalPlan::with_scan_limit(scan, limit)?;
            LogicalPlan::with_scan_limit_hint(scan, limit_hint)
        }
        LogicalPlan::Filter {
            input,
            predicate,
            schema: _,
        } => {
            let mut needed = required_from_above.clone();
            collect_columns(&predicate, &mut needed);
            let pruned_input = prune_with_required(*input, &needed)?;
            LogicalPlan::filter(pruned_input, predicate)
        }
        LogicalPlan::Projection {
            input,
            exprs,
            schema: _,
        } => {
            let needed = exprs_columns(&exprs);
            let pruned_input = prune_with_required(*input, &needed)?;
            LogicalPlan::project(pruned_input, exprs)
        }
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            agg_exprs,
            schema: _,
        } => {
            let mut needed = exprs_columns(&group_exprs);
            for e in &agg_exprs {
                collect_columns(e, &mut needed);
            }
            let pruned_input = prune_with_required(*input, &needed)?;
            LogicalPlan::aggregate(pruned_input, group_exprs, agg_exprs)
        }
        LogicalPlan::Limit {
            input,
            n,
            schema: _,
        } => {
            let pruned_input = prune_with_required(*input, required_from_above)?;
            Ok(LogicalPlan::limit(pruned_input, n))
        }
        LogicalPlan::Join {
            left,
            right,
            on,
            filter,
            join_type,
            schema: out_schema,
        } => {
            // Required set splits across the join.
            //
            // For Semi/Anti the output schema is exactly the left input's,
            // so a parent's required set lives entirely on the left side.
            // For all other join types the output schema is left ++ right
            // (with right-side collision suffixing). Walk the output schema
            // by position: first n_left positions came from left at the
            // same index; the rest came from right at (i - n_left). The
            // output name and the source-side name may differ for the
            // right side because of suffixing; we use the source-side
            // name when registering it as required for the recursive call.
            //
            // On top of parent_required, each side must also keep:
            //   - its half of the join keys
            //   - any columns referenced by the residual filter that
            //     belong to that side's schema
            let left_schema = left.schema();
            let right_schema = right.schema();
            let n_left = left_schema.fields().len();

            let mut left_required: HashSet<String> = HashSet::new();
            let mut right_required: HashSet<String> = HashSet::new();

            if join_type.projects_only_left() {
                // Output is left-only; output names == left input names.
                for f in out_schema.fields() {
                    if required_from_above.contains(f.name()) {
                        left_required.insert(f.name().clone());
                    }
                }
            } else {
                for (i, f) in out_schema.fields().iter().enumerate() {
                    if !required_from_above.contains(f.name()) {
                        continue;
                    }
                    if i < n_left {
                        left_required.insert(left_schema.field(i).name().clone());
                    } else {
                        right_required.insert(right_schema.field(i - n_left).name().clone());
                    }
                }
            }

            for (l, r) in &on {
                collect_columns(l, &mut left_required);
                collect_columns(r, &mut right_required);
            }
            if filter.is_some() {
                // Residual filter exprs reference the *output* schema
                // (post-collision-suffixing). Splitting their column
                // references back to source sides needs an output-name
                // -> (side, source-name) mapping that today's plan node
                // doesn't carry. Conservative fallback: keep every
                // input column on each side. This path is reachable
                // only if a future commit folds both-side preds into
                // Join.filter (Phase 5 does not).
                for f in left_schema.fields() {
                    left_required.insert(f.name().clone());
                }
                for f in right_schema.fields() {
                    right_required.insert(f.name().clone());
                }
            }

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
// Rule: PredicatePushdown  (Week 9 logic, behind the trait)
// =======================================================================

pub struct PredicatePushdown;

impl OptimizerRule for PredicatePushdown {
    fn name(&self) -> &str {
        "predicate_pushdown"
    }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        push_through(plan)
    }
}

fn push_through(plan: LogicalPlan) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Filter {
            input,
            predicate,
            schema: _,
        } => {
            // Filter sitting directly on a Join: split the predicate
            // into AND-conjuncts and route each safely. This is the
            // join-aware pushdown path; its safety rules are the
            // load-bearing piece of the rule (see push_filter_through_join).
            if matches!(*input, LogicalPlan::Join { .. }) {
                return push_filter_through_join(*input, predicate);
            }

            let pushed_input = push_through(*input)?;
            let pushable = collect_pushable(&predicate);
            let new_input = match (pushable.is_empty(), &pushed_input) {
                (true, _) => pushed_input,
                (
                    false,
                    LogicalPlan::Scan {
                        source: ScanSource::Parquet { .. },
                        pushdown_predicates: existing,
                        ..
                    },
                ) => {
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
        LogicalPlan::Projection {
            input,
            exprs,
            schema: _,
        } => {
            let new_input = push_through(*input)?;
            LogicalPlan::project(new_input, exprs)
        }
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            agg_exprs,
            schema: _,
        } => {
            let new_input = push_through(*input)?;
            LogicalPlan::aggregate(new_input, group_exprs, agg_exprs)
        }
        LogicalPlan::Limit {
            input,
            n,
            schema: _,
        } => {
            let new_input = push_through(*input)?;
            Ok(LogicalPlan::limit(new_input, n))
        }
        LogicalPlan::Join {
            left,
            right,
            on,
            filter,
            join_type,
            ..
        } => {
            // Bare Join (no Filter on top): just recurse into both
            // children so further pushdowns inside each subtree run.
            let new_left = push_through(*left)?;
            let new_right = push_through(*right)?;
            LogicalPlan::join(new_left, new_right, on, filter, join_type)
        }
    }
}

/// Handle the `Filter -> Join` shape: split the predicate into AND-conjuncts
/// and route each conjunct based on which side(s) of the join it references
/// and the join type's outer-join safety rules.
///
/// **Safety table** (which conjuncts can be pushed through which join):
///
/// |  Join type   | Left-only | Right-only | Both-side / no-side |
/// |--------------|-----------|------------|---------------------|
/// | Inner / Cross| push left | push right | stay above          |
/// | Left         | push left | stay above | stay above          |
/// | Right        | stay above| push right | stay above          |
/// | Full         | stay above| stay above | stay above          |
/// | Semi / Anti  | push left |    n/a     |   n/a               |
///
/// The "no" cases for outer joins are not stylistic — pushing a right-only
/// predicate through a Left join would discard rows the outer join is
/// supposed to preserve as NULL-padded. The classic counterexample is
/// `right_col IS NULL`: above the Left join it matches the unmatched
/// (NULL-padded) rows; pushed through to the right input it removes those
/// right rows entirely, and the Left-join then produces *more* unmatched
/// rows than the original query (with all-different left values), so the
/// final result diverges. The Phase 3 `left_join_then_filter_on_right_amount_strips_nulls`
/// diff test would catch a regression of this kind end-to-end; the
/// plan-shape tests in this commit catch it at the rule level.
///
/// Both-side conjuncts on Inner could be folded into `Join.filter` (the
/// plan layer accepts residuals on Inner/Cross), which would evaluate the
/// predicate during the probe rather than after. That's a follow-up; for
/// now both-side conjuncts stay above the join.
fn push_filter_through_join(join: LogicalPlan, predicate: Expr) -> Result<LogicalPlan> {
    let LogicalPlan::Join {
        left,
        right,
        on,
        filter,
        join_type,
        ..
    } = join
    else {
        unreachable!("push_filter_through_join called on non-Join input")
    };

    let left_schema = left.schema();
    let right_schema = right.schema();
    let conjuncts = split_conjunction(&predicate);

    let push_left_ok = pushes_into_left_through(join_type);
    let push_right_ok = pushes_into_right_through(join_type);

    let mut push_left: Vec<Expr> = Vec::new();
    let mut push_right: Vec<Expr> = Vec::new();
    let mut keep_above: Vec<Expr> = Vec::new();
    for c in conjuncts {
        let mut cols = HashSet::new();
        collect_columns(&c, &mut cols);
        let refs_left = cols.iter().any(|n| left_schema.field_with_name(n).is_ok());
        let refs_right = cols.iter().any(|n| right_schema.field_with_name(n).is_ok());

        if refs_left && !refs_right && push_left_ok {
            push_left.push(c);
        } else if refs_right && !refs_left && push_right_ok {
            push_right.push(c);
        } else {
            keep_above.push(c);
        }
    }

    // Build the new left subtree: if any conjuncts pushed, wrap left in
    // a Filter and recurse so the wrapped predicate keeps pushing (e.g.
    // into a Parquet scan).
    let new_left = if push_left.is_empty() {
        push_through(*left)?
    } else {
        let merged = combine_and(push_left);
        push_through(LogicalPlan::filter(*left, merged)?)?
    };
    let new_right = if push_right.is_empty() {
        push_through(*right)?
    } else {
        let merged = combine_and(push_right);
        push_through(LogicalPlan::filter(*right, merged)?)?
    };

    let new_join = LogicalPlan::join(new_left, new_right, on, filter, join_type)?;
    if keep_above.is_empty() {
        Ok(new_join)
    } else {
        let merged = combine_and(keep_above);
        LogicalPlan::filter(new_join, merged)
    }
}

fn pushes_into_left_through(jt: plume_core::JoinType) -> bool {
    use plume_core::JoinType::*;
    matches!(jt, Inner | Cross | Left | Semi | Anti)
}
fn pushes_into_right_through(jt: plume_core::JoinType) -> bool {
    use plume_core::JoinType::*;
    matches!(jt, Inner | Cross | Right)
}

/// Split `expr` into its top-level AND-conjuncts. `(a AND b) AND c` ->
/// `[a, b, c]`. Aliased AND nodes are unwrapped. Non-AND exprs return as
/// a single-element vector.
fn split_conjunction(expr: &Expr) -> Vec<Expr> {
    let mut out = Vec::new();
    fn walk(e: &Expr, out: &mut Vec<Expr>) {
        match e {
            Expr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => {
                walk(left, out);
                walk(right, out);
            }
            Expr::Alias { expr, .. } => walk(expr, out),
            other => out.push(other.clone()),
        }
    }
    walk(expr, &mut out);
    out
}

/// Inverse of `split_conjunction`: build `a AND b AND c` from
/// `[a, b, c]`. Single-element input returns the element; empty input
/// is a programmer error (caller must check).
fn combine_and(mut conjuncts: Vec<Expr>) -> Expr {
    debug_assert!(!conjuncts.is_empty());
    let mut acc = conjuncts.remove(0);
    for c in conjuncts {
        acc = Expr::BinaryOp {
            left: Box::new(acc),
            op: BinaryOperator::And,
            right: Box::new(c),
        };
    }
    acc
}

/// The conjuncts of `e` that the scan can use for row-group pruning, in
/// `column OP literal` form.
fn collect_pushable(e: &Expr) -> Vec<Expr> {
    plume_core::conjuncts(e)
        .into_iter()
        .filter_map(plume_core::normalize_pushable)
        .collect()
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
    fn name(&self) -> &str {
        "constant_folding"
    }

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
                    return if *b {
                        r
                    } else {
                        Expr::Literal(ScalarValue::Boolean(false))
                    };
                }
                if let Expr::Literal(ScalarValue::Boolean(b)) = &r {
                    return if *b {
                        l
                    } else {
                        Expr::Literal(ScalarValue::Boolean(false))
                    };
                }
            }
            if op == BinaryOperator::Or {
                if let Expr::Literal(ScalarValue::Boolean(b)) = &l {
                    return if *b {
                        Expr::Literal(ScalarValue::Boolean(true))
                    } else {
                        r
                    };
                }
                if let Expr::Literal(ScalarValue::Boolean(b)) = &r {
                    return if *b {
                        Expr::Literal(ScalarValue::Boolean(true))
                    } else {
                        l
                    };
                }
            }
            // lit OP lit -> lit  (comparisons + arithmetic on matching types)
            if let (Expr::Literal(lv), Expr::Literal(rv)) = (&l, &r) {
                if let Some(folded) = fold_literal_binop(lv, op, rv) {
                    return Expr::Literal(folded);
                }
            }
            Expr::BinaryOp {
                left: Box::new(l),
                op,
                right: Box::new(r),
            }
        }
        Expr::Aggregate { fun, arg } => Expr::Aggregate {
            fun,
            arg: Box::new(fold_expr(*arg)),
        },
        Expr::Alias { expr, name } => Expr::Alias {
            expr: Box::new(fold_expr(*expr)),
            name,
        },
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
        LogicalPlan::Filter {
            input,
            predicate,
            schema: _,
        } => {
            let new_input = rewrite_exprs_in_plan(*input, f)?;
            LogicalPlan::filter(new_input, f(predicate))?
        }
        LogicalPlan::Projection {
            input,
            exprs,
            schema: _,
        } => {
            let new_input = rewrite_exprs_in_plan(*input, f)?;
            let new_exprs = exprs.into_iter().map(f).collect();
            LogicalPlan::project(new_input, new_exprs)?
        }
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            agg_exprs,
            schema: _,
        } => {
            let new_input = rewrite_exprs_in_plan(*input, f)?;
            let new_group = group_exprs.into_iter().map(f).collect();
            let new_agg = agg_exprs.into_iter().map(f).collect();
            LogicalPlan::aggregate(new_input, new_group, new_agg)?
        }
        LogicalPlan::Limit {
            input,
            n,
            schema: _,
        } => {
            let new_input = rewrite_exprs_in_plan(*input, f)?;
            LogicalPlan::limit(new_input, n)
        }
        LogicalPlan::Join {
            left,
            right,
            on,
            filter,
            join_type,
            ..
        } => {
            let new_left = rewrite_exprs_in_plan(*left, f)?;
            let new_right = rewrite_exprs_in_plan(*right, f)?;
            // Apply f to both halves of every key pair, plus the residual
            // filter. f is schema-preserving (it folds expression sub-trees
            // to literals), so the rebuilt Join has the same output schema.
            let new_on: Vec<(Expr, Expr)> = on.into_iter().map(|(l, r)| (f(l), f(r))).collect();
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
    fn name(&self) -> &str {
        "dead_projection_elim"
    }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Projection {
                input,
                exprs,
                schema: _,
            } => {
                let inner = self.rewrite(*input)?;
                if is_identity_projection(&exprs, &inner) {
                    Ok(inner)
                } else {
                    LogicalPlan::project(inner, exprs)
                }
            }
            LogicalPlan::Filter {
                input,
                predicate,
                schema: _,
            } => LogicalPlan::filter(self.rewrite(*input)?, predicate),
            LogicalPlan::Aggregate {
                input,
                group_exprs,
                agg_exprs,
                schema: _,
            } => LogicalPlan::aggregate(self.rewrite(*input)?, group_exprs, agg_exprs),
            LogicalPlan::Limit {
                input,
                n,
                schema: _,
            } => Ok(LogicalPlan::limit(self.rewrite(*input)?, n)),
            LogicalPlan::Scan { .. } => Ok(plan),
            LogicalPlan::Join {
                left,
                right,
                on,
                filter,
                join_type,
                ..
            } => {
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
    fn name(&self) -> &str {
        "limit_pushdown"
    }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Limit {
                input,
                n,
                schema: _,
            } => {
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
            LogicalPlan::Filter {
                input,
                predicate,
                schema: _,
            } => LogicalPlan::filter(self.rewrite(*input)?, predicate),
            LogicalPlan::Projection {
                input,
                exprs,
                schema: _,
            } => LogicalPlan::project(self.rewrite(*input)?, exprs),
            LogicalPlan::Aggregate {
                input,
                group_exprs,
                agg_exprs,
                schema: _,
            } => LogicalPlan::aggregate(self.rewrite(*input)?, group_exprs, agg_exprs),
            LogicalPlan::Scan { .. } => Ok(plan),
            LogicalPlan::Join {
                left,
                right,
                on,
                filter,
                join_type,
                ..
            } => {
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
        LogicalPlan::Scan {
            source,
            schema,
            projection,
            pushdown_predicates,
            limit,
            limit_hint,
        } => {
            // Honour the smaller of any pre-existing scan limit and the new one.
            let merged = match limit {
                None => Some(n),
                Some(existing) => Some(existing.min(n)),
            };
            let scan = LogicalPlan::Scan {
                source,
                schema,
                projection,
                pushdown_predicates,
                limit: merged,
                limit_hint,
            };
            Ok(FoldResult::Folded(scan))
        }
        // Projection preserves row count (one row in, one row out), so we
        // can fold past it: rebuild Projection over a folded scan.
        LogicalPlan::Projection {
            input,
            exprs,
            schema: _,
        } => match try_fold_limit(*input, n)? {
            FoldResult::Folded(new_inner) => {
                Ok(FoldResult::Folded(LogicalPlan::project(new_inner, exprs)?))
            }
            FoldResult::NotFolded(inner) => {
                Ok(FoldResult::NotFolded(LogicalPlan::project(inner, exprs)?))
            }
        },
        // Limit-on-Limit: take the tighter cap and fold through.
        LogicalPlan::Limit {
            input,
            n: inner_n,
            schema: _,
        } => {
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
    format!("BEFORE:\n{}\nAFTER:\n{}", before.explain(), after.explain())
}

// =======================================================================
// Rule: LimitPruning
//
// LimitPushdown won't fold a Limit through a Filter, and it's right not
// to: the scan can't cap rows before the filter. But the scan can still
// choose *which* row groups to read. For
//
//     Limit(k) -> (Projection | Limit)* -> Filter* -> Parquet Scan
//
// this rule sets `limit_hint = k` on the scan. At execute time the scan
// reads fully-matching row groups first and, if they already hold k rows,
// skips everything else (see ParquetScanExec::with_limit_hint).
//
// That's only correct if every row in a FullMatch group passes every
// Filter in the chain, so the rule requires each Filter's predicate to be
// covered by the scan's pushdown predicates. A conjunct the scan never
// saw (e.g. `a < b`) blocks the hint. Projections are allowed only above
// the Filters, so the Filters' column names are the scan's own. Nothing
// else (Aggregate, Join, Sort...) is allowed in the chain: those change
// which rows reach the Limit.
// =======================================================================

pub struct LimitPruning;

impl OptimizerRule for LimitPruning {
    fn name(&self) -> &str {
        "limit_pruning"
    }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Limit { input, n, .. } => {
                let input = match hint_scan(*input, n, &mut Vec::new(), false)? {
                    Ok(hinted) => hinted,
                    Err(unchanged) => self.rewrite(unchanged)?,
                };
                Ok(LogicalPlan::limit(input, n))
            }
            other => map_children(other, |c| self.rewrite(c)),
        }
    }
}

/// Walk `(Projection | Limit)* -> Filter* -> Scan` below a Limit, collecting Filter
/// predicates. Returns `Ok(plan with the hint set)` or `Err(plan unchanged)`
/// when the shape or coverage check fails.
fn hint_scan(
    plan: LogicalPlan,
    k: usize,
    filters: &mut Vec<Expr>,
    seen_filter: bool,
) -> Result<std::result::Result<LogicalPlan, LogicalPlan>> {
    match plan {
        LogicalPlan::Projection { input, exprs, .. } if !seen_filter => {
            Ok(match hint_scan(*input, k, filters, false)? {
                Ok(inner) => Ok(LogicalPlan::project(inner, exprs)?),
                Err(inner) => Err(LogicalPlan::project(inner, exprs)?),
            })
        }
        // A nested Limit above the Filters only tightens how many rows
        // are needed.
        LogicalPlan::Limit { input, n, .. } if !seen_filter => {
            Ok(match hint_scan(*input, k.min(n), filters, false)? {
                Ok(inner) => Ok(LogicalPlan::limit(inner, n)),
                Err(inner) => Err(LogicalPlan::limit(inner, n)),
            })
        }
        LogicalPlan::Filter {
            input, predicate, ..
        } => {
            filters.push(predicate.clone());
            Ok(match hint_scan(*input, k, filters, true)? {
                Ok(inner) => Ok(LogicalPlan::filter(inner, predicate)?),
                Err(inner) => Err(LogicalPlan::filter(inner, predicate)?),
            })
        }
        LogicalPlan::Scan {
            source: ScanSource::Parquet { .. },
            ref pushdown_predicates,
            limit_hint,
            ..
        } => {
            let covered = filters
                .iter()
                .all(|f| plume_core::predicate_covered_by(f, pushdown_predicates));
            if !covered {
                return Ok(Err(plan));
            }
            let hint = Some(limit_hint.map_or(k, |h| h.min(k)));
            Ok(Ok(LogicalPlan::with_scan_limit_hint(plan, hint)?))
        }
        other => Ok(Err(other)),
    }
}

/// Rebuild `plan` with `f` applied to each child.
fn map_children(
    plan: LogicalPlan,
    mut f: impl FnMut(LogicalPlan) -> Result<LogicalPlan>,
) -> Result<LogicalPlan> {
    Ok(match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter {
            input, predicate, ..
        } => LogicalPlan::filter(f(*input)?, predicate)?,
        LogicalPlan::Projection { input, exprs, .. } => LogicalPlan::project(f(*input)?, exprs)?,
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            agg_exprs,
            ..
        } => LogicalPlan::aggregate(f(*input)?, group_exprs, agg_exprs)?,
        LogicalPlan::Limit { input, n, .. } => LogicalPlan::limit(f(*input)?, n),
        LogicalPlan::Join {
            left,
            right,
            on,
            filter,
            join_type,
            ..
        } => LogicalPlan::join(f(*left)?, f(*right)?, on, filter, join_type)?,
    })
}

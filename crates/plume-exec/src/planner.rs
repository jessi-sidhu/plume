//! Logical → physical lowering.
//!
//! For Week 4 the lowering is mechanical (one logical node → one physical
//! operator). The optimizer pass slots in between LogicalPlan and the
//! planner starting Week 8.

use std::sync::Arc;

use arrow_array::RecordBatch;
use plume_core::{predicate_covered_by, Expr, JoinType, LogicalPlan, ScanSource};

use crate::aggregate::{AggregateExec, HashAggregateExec};
use crate::eval::EvalResult;
use crate::exec::ExecutionPlan;
use crate::join::{CrossJoinExec, HashJoinExec};
use crate::join_pruning::{JoinFilter, JoinPruning};
use crate::metrics::{ScanMetrics, ScanMetricsSnapshot};
use crate::operators::{CsvScanExec, FilterExec, LimitExec, ParquetScanExec, ProjectionExec};
use crate::sort::{SortExec, TopKExec};
use crate::topk_pruning::{TopKBoundary, TopKPruning};

/// Execution-time pruning switches. Everything is on by default; the
/// benchmark runner and tests turn a technique off to measure it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerOptions {
    /// Let a TopK share its boundary with the scan below it.
    pub top_k_pruning: bool,
    /// Let a hash join send its build-side key ranges to the probe scan.
    pub join_pruning: bool,
}

impl Default for PlannerOptions {
    fn default() -> Self {
        Self {
            top_k_pruning: true,
            join_pruning: true,
        }
    }
}

/// Runtime pruning handed down from an operator to the Parquet scan it
/// reads from. It passes through Filters, and through Projections that
/// keep the relevant column by name (`Expr::Column`, not an alias that
/// reuses the name). Any other node clears it:
///
/// - a Limit, because top-k pruning reorders the reads and join pruning
///   drops rows, both of which change which rows a Limit keeps;
/// - Aggregate, Sort, TopK and Join, which change what a row means to the
///   operator above.
#[derive(Clone, Default)]
struct Down {
    top_k: Option<TopKPruning>,
    join: Vec<JoinPruning>,
}

impl Down {
    fn through_projection(self, exprs: &[Expr]) -> Down {
        let keeps = |c: &str| exprs.iter().any(|e| matches!(e, Expr::Column(n) if n == c));
        Down {
            top_k: self.top_k.filter(|t| keeps(&t.column)),
            join: self.join.into_iter().filter(|j| keeps(&j.column)).collect(),
        }
    }
}

pub fn create_physical_plan(plan: &LogicalPlan) -> EvalResult<Arc<dyn ExecutionPlan>> {
    build(plan, &mut Vec::new(), PlannerOptions::default())
}

/// A physical plan plus one metrics handle per Parquet scan in it.
pub type PlanWithMetrics = (Arc<dyn ExecutionPlan>, Vec<Arc<ScanMetrics>>);

/// Like `create_physical_plan`, but also returns a metrics handle for
/// every Parquet scan in the physical plan, in depth-first order (a
/// join's left input before its right). Read the handles after the plan's
/// stream has been drained.
pub fn create_physical_plan_with_metrics(plan: &LogicalPlan) -> EvalResult<PlanWithMetrics> {
    create_physical_plan_with_options(plan, PlannerOptions::default())
}

/// `create_physical_plan_with_metrics` with explicit pruning switches.
pub fn create_physical_plan_with_options(
    plan: &LogicalPlan,
    options: PlannerOptions,
) -> EvalResult<PlanWithMetrics> {
    let mut metrics = Vec::new();
    let exec = build(plan, &mut metrics, options)?;
    Ok((exec, metrics))
}

fn build(
    plan: &LogicalPlan,
    metrics: &mut Vec<Arc<ScanMetrics>>,
    options: PlannerOptions,
) -> EvalResult<Arc<dyn ExecutionPlan>> {
    build_with(plan, metrics, options, Down::default())
}

/// `build`, with runtime pruning `down` waiting to reach a scan.
fn build_with(
    plan: &LogicalPlan,
    metrics: &mut Vec<Arc<ScanMetrics>>,
    options: PlannerOptions,
    down: Down,
) -> EvalResult<Arc<dyn ExecutionPlan>> {
    Ok(match plan {
        LogicalPlan::Scan {
            source,
            schema,
            projection,
            pushdown_predicates,
            limit,
            limit_hint,
        } => match source {
            ScanSource::Csv { path } => {
                // CSV: pruning + pushdown don't apply; limit does (cheap to
                // truncate the iterator). Build the plain CsvScanExec, wrap
                // in LimitExec downstream if needed via the Limit operator.
                let _ = projection;
                let _ = pushdown_predicates;
                let _ = limit;
                Arc::new(CsvScanExec::new(path.clone(), schema.clone()))
            }
            ScanSource::Parquet { path } => {
                let mut exec = match projection {
                    None => ParquetScanExec::new(path.clone(), schema.clone()),
                    Some(cols) => {
                        ParquetScanExec::with_projection(path.clone(), schema.clone(), cols.clone())
                    }
                };
                if !pushdown_predicates.is_empty() {
                    exec = exec.with_pushdown_predicates(pushdown_predicates.clone());
                }
                if let Some(n) = limit {
                    exec = exec.with_limit(*n);
                }
                if let Some(k) = limit_hint {
                    exec = exec.with_limit_hint(*k);
                }
                if let Some(t) = down.top_k {
                    if schema.field_with_name(&t.column).is_ok() {
                        exec = exec.with_top_k_pruning(t);
                    }
                }
                for j in down.join {
                    if schema.field_with_name(&j.column).is_ok() {
                        exec = exec.with_join_pruning(j);
                    }
                }
                metrics.push(exec.metrics());
                Arc::new(exec)
            }
        },
        LogicalPlan::Filter {
            input, predicate, ..
        } => {
            let child = build_with(input, metrics, options, down)?;
            let mut filter = FilterExec::new(child, predicate.clone());
            // A Filter directly on a Parquet scan can pass FullMatch batches
            // through unevaluated, but only if the scan's pushdown
            // predicates cover every conjunct of the filter. Otherwise a
            // FullMatch group (judged on the pushed conjuncts alone) could
            // still hold rows the remaining conjuncts reject.
            if let LogicalPlan::Scan {
                source: ScanSource::Parquet { .. },
                pushdown_predicates,
                ..
            } = input.as_ref()
            {
                if !pushdown_predicates.is_empty()
                    && predicate_covered_by(predicate, pushdown_predicates)
                {
                    // The child was the last Parquet scan built.
                    if let Some(scan) = metrics.last() {
                        filter = filter.with_full_match_skipping(Arc::clone(scan));
                    }
                }
            }
            Arc::new(filter)
        }
        LogicalPlan::Projection {
            input,
            exprs,
            schema,
        } => {
            let child = build_with(input, metrics, options, down.through_projection(exprs))?;
            Arc::new(ProjectionExec::new(child, exprs.clone(), schema.clone()))
        }
        LogicalPlan::Limit { input, n, .. } => {
            let child = build(input, metrics, options)?;
            Arc::new(LimitExec::new(child, *n))
        }
        LogicalPlan::Sort { input, exprs, .. } => {
            let child = build(input, metrics, options)?;
            Arc::new(SortExec::new(child, exprs))
        }
        LogicalPlan::TopK {
            input, exprs, k, ..
        } => match &exprs[0].expr {
            Expr::Column(column) if options.top_k_pruning => {
                let boundary = Arc::new(TopKBoundary::default());
                let pruning = TopKPruning {
                    column: column.clone(),
                    asc: exprs[0].asc,
                    nulls_first: exprs[0].nulls_first,
                    boundary: Arc::clone(&boundary),
                };
                let down = Down {
                    top_k: Some(pruning),
                    join: Vec::new(),
                };
                let child = build_with(input, metrics, options, down)?;
                Arc::new(TopKExec::new(child, exprs, *k).with_boundary(boundary))
            }
            _ => {
                let child = build(input, metrics, options)?;
                Arc::new(TopKExec::new(child, exprs, *k))
            }
        },
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            agg_exprs,
            schema,
        } => {
            let child = build(input, metrics, options)?;
            if group_exprs.is_empty() {
                Arc::new(AggregateExec::new(child, agg_exprs.clone(), schema.clone()))
            } else {
                Arc::new(HashAggregateExec::new(
                    child,
                    group_exprs.clone(),
                    agg_exprs.clone(),
                    schema.clone(),
                ))
            }
        }
        LogicalPlan::Join {
            left,
            right,
            on,
            filter,
            join_type,
            schema,
        } => {
            // Join pruning: for Inner, Semi and Right joins (which drop
            // probe rows without a match), each key that is a plain probe
            // column gets a filter the build side fills in.
            let mut probe_down = Down::default();
            let mut dynamic_filters = Vec::new();
            if options.join_pruning
                && matches!(
                    join_type,
                    JoinType::Inner | JoinType::Semi | JoinType::Right
                )
            {
                for (i, (l, _)) in on.iter().enumerate() {
                    if let Expr::Column(c) = l {
                        let filter = Arc::new(JoinFilter::default());
                        probe_down.join.push(JoinPruning {
                            column: c.clone(),
                            filter: Arc::clone(&filter),
                        });
                        dynamic_filters.push((i, filter));
                    }
                }
            }
            let left_exec = build_with(left, metrics, options, probe_down)?;
            let right_exec = build(right, metrics, options)?;
            match join_type {
                JoinType::Cross => Arc::new(CrossJoinExec::new(
                    left_exec,
                    right_exec,
                    filter.clone(),
                    schema.clone(),
                )),
                JoinType::Inner
                | JoinType::Left
                | JoinType::Right
                | JoinType::Full
                | JoinType::Semi
                | JoinType::Anti => Arc::new(
                    HashJoinExec::new(
                        left_exec,
                        right_exec,
                        on.clone(),
                        filter.clone(),
                        *join_type,
                        schema.clone(),
                    )
                    .with_dynamic_filters(dynamic_filters),
                ),
            }
        }
    })
}

/// Run a logical plan to completion and return all output batches.
/// Convenience for the end-of-pipeline `.collect()` ergonomics.
pub fn collect(plan: &LogicalPlan) -> EvalResult<Vec<RecordBatch>> {
    let exec = create_physical_plan(plan)?;
    exec.execute()?.collect::<EvalResult<Vec<_>>>()
}

/// `collect`, plus a snapshot of every Parquet scan's metrics (same order
/// as `create_physical_plan_with_metrics`), taken after the stream is
/// drained.
pub fn collect_with_metrics(
    plan: &LogicalPlan,
) -> EvalResult<(Vec<RecordBatch>, Vec<ScanMetricsSnapshot>)> {
    collect_with_options(plan, PlannerOptions::default())
}

/// `collect_with_metrics` with explicit pruning switches.
pub fn collect_with_options(
    plan: &LogicalPlan,
    options: PlannerOptions,
) -> EvalResult<(Vec<RecordBatch>, Vec<ScanMetricsSnapshot>)> {
    let (exec, metrics) = create_physical_plan_with_options(plan, options)?;
    let batches = exec.execute()?.collect::<EvalResult<Vec<_>>>()?;
    Ok((batches, metrics.iter().map(|m| m.snapshot()).collect()))
}

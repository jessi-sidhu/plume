//! Logical → physical lowering.
//!
//! For Week 4 the lowering is mechanical (one logical node → one physical
//! operator). The optimizer pass slots in between LogicalPlan and the
//! planner starting Week 8.

use std::sync::Arc;

use arrow_array::RecordBatch;
use plume_core::{predicate_covered_by, JoinType, LogicalPlan, ScanSource};

use crate::aggregate::{AggregateExec, HashAggregateExec};
use crate::eval::EvalResult;
use crate::exec::ExecutionPlan;
use crate::join::{CrossJoinExec, HashJoinExec};
use crate::metrics::{ScanMetrics, ScanMetricsSnapshot};
use crate::operators::{CsvScanExec, FilterExec, LimitExec, ParquetScanExec, ProjectionExec};

pub fn create_physical_plan(plan: &LogicalPlan) -> EvalResult<Arc<dyn ExecutionPlan>> {
    build(plan, &mut Vec::new())
}

/// A physical plan plus one metrics handle per Parquet scan in it.
pub type PlanWithMetrics = (Arc<dyn ExecutionPlan>, Vec<Arc<ScanMetrics>>);

/// Like `create_physical_plan`, but also returns a metrics handle for
/// every Parquet scan in the physical plan, in depth-first order (a
/// join's left input before its right). Read the handles after the plan's
/// stream has been drained.
pub fn create_physical_plan_with_metrics(plan: &LogicalPlan) -> EvalResult<PlanWithMetrics> {
    let mut metrics = Vec::new();
    let exec = build(plan, &mut metrics)?;
    Ok((exec, metrics))
}

fn build(
    plan: &LogicalPlan,
    metrics: &mut Vec<Arc<ScanMetrics>>,
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
                metrics.push(exec.metrics());
                Arc::new(exec)
            }
        },
        LogicalPlan::Filter {
            input, predicate, ..
        } => {
            let child = build(input, metrics)?;
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
            let child = build(input, metrics)?;
            Arc::new(ProjectionExec::new(child, exprs.clone(), schema.clone()))
        }
        LogicalPlan::Limit { input, n, .. } => {
            let child = build(input, metrics)?;
            Arc::new(LimitExec::new(child, *n))
        }
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            agg_exprs,
            schema,
        } => {
            let child = build(input, metrics)?;
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
            let left_exec = build(left, metrics)?;
            let right_exec = build(right, metrics)?;
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
                | JoinType::Anti => Arc::new(HashJoinExec::new(
                    left_exec,
                    right_exec,
                    on.clone(),
                    filter.clone(),
                    *join_type,
                    schema.clone(),
                )),
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
    let (exec, metrics) = create_physical_plan_with_metrics(plan)?;
    let batches = exec.execute()?.collect::<EvalResult<Vec<_>>>()?;
    Ok((batches, metrics.iter().map(|m| m.snapshot()).collect()))
}

//! Logical → physical lowering.
//!
//! For Week 4 the lowering is mechanical (one logical node → one physical
//! operator). The optimizer pass slots in between LogicalPlan and the
//! planner starting Week 8.

use std::sync::Arc;

use arrow_array::RecordBatch;
use plume_core::{LogicalPlan, ScanSource};

use crate::aggregate::{AggregateExec, HashAggregateExec};
use crate::eval::EvalResult;
use crate::exec::ExecutionPlan;
use crate::operators::{CsvScanExec, FilterExec, LimitExec, ParquetScanExec, ProjectionExec};

pub fn create_physical_plan(plan: &LogicalPlan) -> EvalResult<Arc<dyn ExecutionPlan>> {
    Ok(match plan {
        LogicalPlan::Scan { source, schema, projection, pushdown_predicates, limit } => match source {
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
                    Some(cols) => ParquetScanExec::with_projection(
                        path.clone(),
                        schema.clone(),
                        cols.clone(),
                    ),
                };
                if !pushdown_predicates.is_empty() {
                    exec = exec.with_pushdown_predicates(pushdown_predicates.clone());
                }
                if let Some(n) = limit {
                    exec = exec.with_limit(*n);
                }
                Arc::new(exec)
            }
        },
        LogicalPlan::Filter { input, predicate, .. } => {
            let child = create_physical_plan(input)?;
            Arc::new(FilterExec::new(child, predicate.clone()))
        }
        LogicalPlan::Projection { input, exprs, schema } => {
            let child = create_physical_plan(input)?;
            Arc::new(ProjectionExec::new(child, exprs.clone(), schema.clone()))
        }
        LogicalPlan::Limit { input, n, .. } => {
            let child = create_physical_plan(input)?;
            Arc::new(LimitExec::new(child, *n))
        }
        LogicalPlan::Aggregate { input, group_exprs, agg_exprs, schema } => {
            let child = create_physical_plan(input)?;
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
    })
}

/// Run a logical plan to completion and return all output batches.
/// Convenience for the end-of-pipeline `.collect()` ergonomics.
pub fn collect(plan: &LogicalPlan) -> EvalResult<Vec<RecordBatch>> {
    let exec = create_physical_plan(plan)?;
    exec.execute()?.collect::<EvalResult<Vec<_>>>()
}

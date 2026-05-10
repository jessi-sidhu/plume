//! DataFrame-style builder. Construct a `LogicalPlan` by chaining methods,
//! then hand it to the executor (Week 4+) via `into_plan()`. No execution
//! happens here; this layer is pure tree construction.

use std::path::PathBuf;

use arrow_schema::SchemaRef;

use crate::expr::Expr;
use crate::plan::{JoinType, LogicalPlan};
use crate::Result;

#[derive(Debug, Clone)]
pub struct DataFrame {
    plan: LogicalPlan,
}

impl DataFrame {
    pub fn scan_csv(path: impl Into<PathBuf>, schema: SchemaRef) -> Self {
        DataFrame { plan: LogicalPlan::scan_csv(path, schema) }
    }

    pub fn scan_parquet(path: impl Into<PathBuf>, schema: SchemaRef) -> Self {
        DataFrame { plan: LogicalPlan::scan_parquet(path, schema) }
    }

    pub fn filter(self, predicate: Expr) -> Result<Self> {
        Ok(DataFrame { plan: LogicalPlan::filter(self.plan, predicate)? })
    }

    pub fn project(self, exprs: Vec<Expr>) -> Result<Self> {
        Ok(DataFrame { plan: LogicalPlan::project(self.plan, exprs)? })
    }

    pub fn aggregate(self, group: Vec<Expr>, aggs: Vec<Expr>) -> Result<Self> {
        Ok(DataFrame { plan: LogicalPlan::aggregate(self.plan, group, aggs)? })
    }

    pub fn limit(self, n: usize) -> Self {
        DataFrame { plan: LogicalPlan::limit(self.plan, n) }
    }

    /// Equi-join `self` with `right` on `on` (parallel `(left_expr,
    /// right_expr)` pairs). Use one of the `JoinType` variants except
    /// `Cross` — pass an empty `on` and `JoinType::Cross` (or use
    /// `cross_join`) for the no-keys case.
    pub fn join(
        self,
        right: DataFrame,
        on: Vec<(Expr, Expr)>,
        join_type: JoinType,
    ) -> Result<Self> {
        Ok(DataFrame {
            plan: LogicalPlan::join(self.plan, right.plan, on, None, join_type)?,
        })
    }

    /// Cross join: every row on the left paired with every row on the right.
    pub fn cross_join(self, right: DataFrame) -> Result<Self> {
        Ok(DataFrame { plan: LogicalPlan::cross_join(self.plan, right.plan)? })
    }

    pub fn schema(&self) -> SchemaRef {
        self.plan.schema()
    }

    pub fn explain(&self) -> String {
        self.plan.explain()
    }

    pub fn into_plan(self) -> LogicalPlan {
        self.plan
    }

    pub fn plan(&self) -> &LogicalPlan {
        &self.plan
    }
}

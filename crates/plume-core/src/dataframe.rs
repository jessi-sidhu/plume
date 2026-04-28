//! DataFrame-style builder. Construct a `LogicalPlan` by chaining methods,
//! then hand it to the executor (Week 4+) via `into_plan()`. No execution
//! happens here; this layer is pure tree construction.

use std::path::PathBuf;

use arrow_schema::SchemaRef;

use crate::expr::Expr;
use crate::plan::LogicalPlan;
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

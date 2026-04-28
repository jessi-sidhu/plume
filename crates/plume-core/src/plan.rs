//! Logical query plan + schema propagation.
//!
//! Each variant carries the *output* schema as `SchemaRef` so callers can
//! inspect it without re-walking the tree. Schema propagation rules live in
//! the constructors (`Filter::try_new`, `Projection::try_new`, etc.) — once a
//! plan exists, its schema is trustworthy.

use std::fmt::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_schema::{DataType, Schema, SchemaRef};

use crate::expr::{fields_for, Expr};
use crate::{PlumeError, Result};

#[derive(Debug, Clone)]
pub enum ScanSource {
    /// CSV scans need a caller-supplied schema (CSV files have no embedded
    /// type info). Inference is a separate concern, deliberately left out.
    Csv { path: PathBuf },
    /// Parquet scans *could* read schema from file metadata, but doing so
    /// requires an I/O dependency this crate doesn't pull in. For Week 2 the
    /// caller still supplies the schema; Week 7 introduces an
    /// `auto_scan_parquet` helper in `plume-io`.
    Parquet { path: PathBuf },
}

impl ScanSource {
    pub fn display_path(&self) -> String {
        match self {
            ScanSource::Csv { path } | ScanSource::Parquet { path } => path.display().to_string(),
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            ScanSource::Csv { .. } => "CsvScan",
            ScanSource::Parquet { .. } => "ParquetScan",
        }
    }
}

#[derive(Debug, Clone)]
pub enum LogicalPlan {
    Scan {
        source: ScanSource,
        /// Output schema. When `projection` is `None` this is the full file
        /// schema; when `Some(cols)` it is the file schema restricted to and
        /// reordered by `cols`. Always reflects what downstream operators see.
        schema: SchemaRef,
        /// Set by the column-pruning optimizer (Week 8). `None` = read every
        /// column. `Some(names)` = read just these, in this order.
        projection: Option<Vec<String>>,
    },
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
        // Filter preserves input schema; cached for cheap `.schema()`.
        schema: SchemaRef,
    },
    Projection {
        input: Box<LogicalPlan>,
        exprs: Vec<Expr>,
        schema: SchemaRef,
    },
    Aggregate {
        input: Box<LogicalPlan>,
        group_exprs: Vec<Expr>,
        agg_exprs: Vec<Expr>,
        schema: SchemaRef,
    },
    Limit {
        input: Box<LogicalPlan>,
        n: usize,
        schema: SchemaRef,
    },
}

impl LogicalPlan {
    pub fn schema(&self) -> SchemaRef {
        match self {
            LogicalPlan::Scan { schema, .. }
            | LogicalPlan::Filter { schema, .. }
            | LogicalPlan::Projection { schema, .. }
            | LogicalPlan::Aggregate { schema, .. }
            | LogicalPlan::Limit { schema, .. } => schema.clone(),
        }
    }

    // ---- constructors that enforce schema propagation rules ---------------

    pub fn scan_csv(path: impl Into<PathBuf>, schema: SchemaRef) -> Self {
        LogicalPlan::Scan {
            source: ScanSource::Csv { path: path.into() },
            schema,
            projection: None,
        }
    }

    pub fn scan_parquet(path: impl Into<PathBuf>, schema: SchemaRef) -> Self {
        LogicalPlan::Scan {
            source: ScanSource::Parquet { path: path.into() },
            schema,
            projection: None,
        }
    }

    /// Construct a Scan with column projection already applied. `schema` must
    /// be the full file schema; we restrict it to and reorder it by `cols`.
    /// Used by the column-pruning optimizer.
    pub fn scan_with_projection(
        source: ScanSource,
        full_schema: SchemaRef,
        cols: Vec<String>,
    ) -> Result<Self> {
        let mut fields = Vec::with_capacity(cols.len());
        for name in &cols {
            let f = full_schema
                .field_with_name(name)
                .map_err(|_| PlumeError::Schema(format!(
                    "scan_with_projection: column `{name}` not in source schema"
                )))?;
            fields.push(f.clone());
        }
        let projected_schema = Arc::new(Schema::new(fields));
        Ok(LogicalPlan::Scan {
            source,
            schema: projected_schema,
            projection: Some(cols),
        })
    }

    pub fn filter(input: LogicalPlan, predicate: Expr) -> Result<Self> {
        let in_schema = input.schema();
        let pred_ty = predicate.data_type(&in_schema)?;
        if !matches!(pred_ty, DataType::Boolean | DataType::Null) {
            return Err(PlumeError::Type(format!(
                "filter predicate must be Boolean, got {pred_ty:?}"
            )));
        }
        // Filter preserves input schema (same fields, same nullability —
        // filter cannot make a non-null column nullable).
        Ok(LogicalPlan::Filter { schema: in_schema.clone(), input: Box::new(input), predicate })
    }

    pub fn project(input: LogicalPlan, exprs: Vec<Expr>) -> Result<Self> {
        if exprs.is_empty() {
            return Err(PlumeError::Plan("projection requires at least one expression".into()));
        }
        let schema = fields_for(&input.schema(), &exprs)?;
        Ok(LogicalPlan::Projection { input: Box::new(input), exprs, schema })
    }

    pub fn aggregate(
        input: LogicalPlan,
        group_exprs: Vec<Expr>,
        agg_exprs: Vec<Expr>,
    ) -> Result<Self> {
        if group_exprs.is_empty() && agg_exprs.is_empty() {
            return Err(PlumeError::Plan(
                "aggregate requires at least one group or aggregate expression".into(),
            ));
        }
        // All agg_exprs must actually be aggregates — catch
        // `aggregate(.., vec![col("x")])` here, not at execution time.
        for e in &agg_exprs {
            if !is_aggregate_expr(e) {
                return Err(PlumeError::Plan(format!(
                    "aggregate expression `{}` is not an aggregate function",
                    e.display_name()
                )));
            }
        }
        let in_schema = input.schema();
        let group_fields = fields_for(&in_schema, &group_exprs)?;
        let agg_fields = fields_for(&in_schema, &agg_exprs)?;
        // Output: group columns first (preserves user-visible column order),
        // then aggregate columns. Combined into one schema.
        let mut combined = Vec::with_capacity(group_fields.fields().len() + agg_fields.fields().len());
        combined.extend(group_fields.fields().iter().map(|f| (**f).clone()));
        combined.extend(agg_fields.fields().iter().map(|f| (**f).clone()));
        let schema = Arc::new(Schema::new(combined));
        Ok(LogicalPlan::Aggregate { input: Box::new(input), group_exprs, agg_exprs, schema })
    }

    pub fn limit(input: LogicalPlan, n: usize) -> Self {
        let schema = input.schema();
        LogicalPlan::Limit { input: Box::new(input), n, schema }
    }

    pub fn explain(&self) -> String {
        let mut s = String::new();
        write_indented(self, &mut s, 0).expect("write to String never fails");
        s
    }
}

fn is_aggregate_expr(e: &Expr) -> bool {
    match e {
        Expr::Aggregate { .. } => true,
        Expr::Alias { expr, .. } => is_aggregate_expr(expr),
        _ => false,
    }
}

fn write_indented(plan: &LogicalPlan, out: &mut String, depth: usize) -> fmt::Result {
    let pad = "  ".repeat(depth);
    match plan {
        LogicalPlan::Scan { source, schema, projection } => {
            match projection {
                None => writeln!(
                    out,
                    "{pad}{}: {} [{} cols]",
                    source.kind(),
                    source.display_path(),
                    schema.fields().len()
                )?,
                Some(cols) => writeln!(
                    out,
                    "{pad}{}: {}, projection=[{}]",
                    source.kind(),
                    source.display_path(),
                    cols.join(", ")
                )?,
            }
        }
        LogicalPlan::Filter { input, predicate, .. } => {
            writeln!(out, "{pad}Filter: {}", predicate.display_name())?;
            write_indented(input, out, depth + 1)?;
        }
        LogicalPlan::Projection { input, exprs, .. } => {
            let names: Vec<String> = exprs.iter().map(|e| e.display_name()).collect();
            writeln!(out, "{pad}Projection: [{}]", names.join(", "))?;
            write_indented(input, out, depth + 1)?;
        }
        LogicalPlan::Aggregate { input, group_exprs, agg_exprs, .. } => {
            let g: Vec<String> = group_exprs.iter().map(|e| e.display_name()).collect();
            let a: Vec<String> = agg_exprs.iter().map(|e| e.display_name()).collect();
            writeln!(out, "{pad}Aggregate: groupBy=[{}] aggs=[{}]", g.join(", "), a.join(", "))?;
            write_indented(input, out, depth + 1)?;
        }
        LogicalPlan::Limit { input, n, .. } => {
            writeln!(out, "{pad}Limit: {n}")?;
            write_indented(input, out, depth + 1)?;
        }
    }
    Ok(())
}

impl fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.explain())
    }
}

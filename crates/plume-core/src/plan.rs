//! Logical query plan + schema propagation.
//!
//! Each variant carries the *output* schema as `SchemaRef` so callers can
//! inspect it without re-walking the tree. Schema propagation rules live in
//! the constructors (`Filter::try_new`, `Projection::try_new`, etc.) — once a
//! plan exists, its schema is trustworthy.

use std::fmt::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::expr::{fields_for, Expr};
use crate::{PlumeError, Result};

/// Equi-join semantics. Cross is the no-keys nested-loop case; Semi/Anti
/// emit only the left side (right is consulted but not projected).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Semi,
    Anti,
    Cross,
}

impl JoinType {
    pub fn name(self) -> &'static str {
        match self {
            JoinType::Inner => "Inner",
            JoinType::Left => "Left",
            JoinType::Right => "Right",
            JoinType::Full => "Full",
            JoinType::Semi => "Semi",
            JoinType::Anti => "Anti",
            JoinType::Cross => "Cross",
        }
    }
    /// True if this join may emit NULL-padded rows from the *left* input
    /// for which there was no right-side match. Drives schema nullability
    /// and (later) optimizer pushdown safety.
    pub fn left_is_preserved(self) -> bool {
        matches!(self, JoinType::Left | JoinType::Full)
    }
    /// True if this join may emit NULL-padded rows from the *right* input
    /// for which there was no left-side match.
    pub fn right_is_preserved(self) -> bool {
        matches!(self, JoinType::Right | JoinType::Full)
    }
    /// True if the join's output schema only contains the left side
    /// (Semi / Anti). Right columns are used for matching, never emitted.
    pub fn projects_only_left(self) -> bool {
        matches!(self, JoinType::Semi | JoinType::Anti)
    }
}

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
        /// Conjunction of predicates pushed in by the predicate-pushdown
        /// optimizer (Week 9). The scan uses these as a *hint* to skip row
        /// groups whose statistics rule out a match. The Filter that
        /// originally held them stays in the plan and still evaluates the
        /// real per-row check — pushdown is a scan-level elision, not a
        /// replacement for filtering.
        pushdown_predicates: Vec<Expr>,
        /// Set by limit pushdown (Week 10). When `Some(n)`, the scan reads
        /// at most `n` rows from the source. The Limit operator above
        /// stays in the plan to enforce the contract end-to-end.
        limit: Option<usize>,
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
    /// Equi-join (or Cross when `on` is empty). The first multi-input plan
    /// node in the engine. Schema is computed at construction time and
    /// cached: it concatenates left then right with collision-suffixing
    /// (a column named `id` on both sides becomes `id` from the left and
    /// `id_1` from the right), with nullability widened for any side the
    /// join may emit unmatched rows for. Semi/Anti project only the left
    /// schema verbatim.
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        /// Equi-join key pairs. Empty iff `join_type == Cross`. Each
        /// `(l, r)` is type-checked against the respective input schemas
        /// at construction time; `l` resolves in `left.schema`, `r` in
        /// `right.schema`, and the two output types must match.
        on: Vec<(Expr, Expr)>,
        /// Optional residual predicate evaluated against the concatenated
        /// row. Used by the optimizer to fold both-side predicates that
        /// can't be pushed below the join (inner / cross only — outer
        /// joins don't accept residuals here, see PredicatePushdown).
        filter: Option<Expr>,
        join_type: JoinType,
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
            | LogicalPlan::Limit { schema, .. }
            | LogicalPlan::Join { schema, .. } => schema.clone(),
        }
    }

    // ---- constructors that enforce schema propagation rules ---------------

    pub fn scan_csv(path: impl Into<PathBuf>, schema: SchemaRef) -> Self {
        LogicalPlan::Scan {
            source: ScanSource::Csv { path: path.into() },
            schema,
            projection: None,
            pushdown_predicates: Vec::new(),
            limit: None,
        }
    }

    pub fn scan_parquet(path: impl Into<PathBuf>, schema: SchemaRef) -> Self {
        LogicalPlan::Scan {
            source: ScanSource::Parquet { path: path.into() },
            schema,
            projection: None,
            pushdown_predicates: Vec::new(),
            limit: None,
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
            pushdown_predicates: Vec::new(),
            limit: None,
        })
    }

    /// Attach pushdown predicates to a Scan node. Preserves any existing
    /// projection / limit. Used by the predicate-pushdown optimizer (Week 9).
    pub fn with_pushdown_predicates(plan: LogicalPlan, predicates: Vec<Expr>) -> Result<Self> {
        match plan {
            LogicalPlan::Scan { source, schema, projection, limit, .. } => Ok(LogicalPlan::Scan {
                source,
                schema,
                projection,
                pushdown_predicates: predicates,
                limit,
            }),
            _ => Err(PlumeError::Plan(
                "with_pushdown_predicates: target plan must be a Scan".into(),
            )),
        }
    }

    /// Set (or clear) a row limit on a Scan node. Used by limit pushdown
    /// (Week 10).
    pub fn with_scan_limit(plan: LogicalPlan, limit: Option<usize>) -> Result<Self> {
        match plan {
            LogicalPlan::Scan { source, schema, projection, pushdown_predicates, .. } => {
                Ok(LogicalPlan::Scan {
                    source,
                    schema,
                    projection,
                    pushdown_predicates,
                    limit,
                })
            }
            _ => Err(PlumeError::Plan(
                "with_scan_limit: target plan must be a Scan".into(),
            )),
        }
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

    /// Construct a Join plan node. Validates each `(l, r)` key pair against
    /// the corresponding input schema and computes the output schema with
    /// collision suffixing + nullability widening. `Cross` requires `on` to
    /// be empty; every other variant requires `on` to be non-empty.
    pub fn join(
        left: LogicalPlan,
        right: LogicalPlan,
        on: Vec<(Expr, Expr)>,
        filter: Option<Expr>,
        join_type: JoinType,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();

        // Key shape rules.
        match join_type {
            JoinType::Cross => {
                if !on.is_empty() {
                    return Err(PlumeError::Plan(
                        "cross join takes no equi-keys".into(),
                    ));
                }
            }
            _ => {
                if on.is_empty() {
                    return Err(PlumeError::Plan(format!(
                        "{} join requires at least one equi-key",
                        join_type.name()
                    )));
                }
            }
        }

        // Type-check each key pair: l resolves in left, r resolves in right,
        // and their resolved types must match (same comparability rule the
        // engine applies to BinaryOp::Eq).
        for (l, r) in &on {
            let lt = l.data_type(&left_schema)?;
            let rt = r.data_type(&right_schema)?;
            if !types_equi_compatible(&lt, &rt) {
                return Err(PlumeError::Type(format!(
                    "join key types must match: left `{}` is {:?}, right `{}` is {:?}",
                    l.display_name(),
                    lt,
                    r.display_name(),
                    rt
                )));
            }
        }

        // Optional residual: must be Boolean and resolve against the
        // *concatenated* schema. We build a temporary concat schema with
        // the same disambiguation rules, then ask the predicate to
        // type-check against it.
        let concat_for_filter = concat_schema(&left_schema, &right_schema, join_type)?;
        if let Some(f) = &filter {
            let ft = f.data_type(&concat_for_filter)?;
            if !matches!(ft, DataType::Boolean | DataType::Null) {
                return Err(PlumeError::Type(format!(
                    "join filter must be Boolean, got {ft:?}"
                )));
            }
            // Outer joins don't accept residuals here. They turn into
                // semantics-changing pushdowns; the optimizer is responsible
                // for not folding both-side predicates into outer-join
                // residuals. Catch caller mistakes early.
            if matches!(
                join_type,
                JoinType::Left | JoinType::Right | JoinType::Full
            ) {
                return Err(PlumeError::Plan(format!(
                    "{} join does not accept a residual filter; \
                     wrap the join in an outer Filter instead",
                    join_type.name()
                )));
            }
        }

        let schema = concat_for_filter;

        Ok(LogicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
            on,
            filter,
            join_type,
            schema,
        })
    }

    /// Convenience: cross-join with no equi-keys and no residual.
    pub fn cross_join(left: LogicalPlan, right: LogicalPlan) -> Result<Self> {
        Self::join(left, right, Vec::new(), None, JoinType::Cross)
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

/// Mirror of the `Eq`/`NotEq` comparability rule from `expr::resolve_binary`.
/// Same type matches; either-Null matches; otherwise reject.
fn types_equi_compatible(l: &DataType, r: &DataType) -> bool {
    if l == r {
        return true;
    }
    matches!(l, DataType::Null) || matches!(r, DataType::Null)
}

/// Concatenate two input schemas into the join's output schema.
///
/// Rules (matches the project's design decision recorded in the joins plan):
/// - Semi/Anti: output is exactly the left schema.
/// - Otherwise: left fields first, then right fields. Right-side names that
///   collide with a left-side name (or with a previously-emitted right name)
///   are suffixed `_1`, `_2`, ... until unique.
/// - Nullability widens for the side the join may emit unmatched-row NULLs
///   for: Right/Full make left columns nullable; Left/Full make right
///   columns nullable. (Inner / Cross preserve input nullability.)
fn concat_schema(
    left: &Schema,
    right: &Schema,
    join_type: JoinType,
) -> Result<SchemaRef> {
    if join_type.projects_only_left() {
        return Ok(Arc::new(left.clone()));
    }

    let widen_left = join_type.right_is_preserved();
    let widen_right = join_type.left_is_preserved();

    let mut fields: Vec<Field> = Vec::with_capacity(left.fields().len() + right.fields().len());
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();

    for f in left.fields() {
        let nullable = f.is_nullable() || widen_left;
        let field = Field::new(f.name(), f.data_type().clone(), nullable);
        used.insert(field.name().clone());
        fields.push(field);
    }
    for f in right.fields() {
        let nullable = f.is_nullable() || widen_right;
        let name = disambiguate(f.name(), &used);
        used.insert(name.clone());
        fields.push(Field::new(&name, f.data_type().clone(), nullable));
    }

    Ok(Arc::new(Schema::new(fields)))
}

/// Pick the next free name in the `name`, `name_1`, `name_2`, ... sequence.
fn disambiguate(name: &str, used: &std::collections::HashSet<String>) -> String {
    if !used.contains(name) {
        return name.to_string();
    }
    let mut i = 1usize;
    loop {
        let candidate = format!("{name}_{i}");
        if !used.contains(&candidate) {
            return candidate;
        }
        i += 1;
    }
}

fn write_indented(plan: &LogicalPlan, out: &mut String, depth: usize) -> fmt::Result {
    let pad = "  ".repeat(depth);
    match plan {
        LogicalPlan::Scan { source, schema, projection, pushdown_predicates, limit } => {
            match projection {
                None => write!(
                    out,
                    "{pad}{}: {} [{} cols]",
                    source.kind(),
                    source.display_path(),
                    schema.fields().len()
                )?,
                Some(cols) => write!(
                    out,
                    "{pad}{}: {}, projection=[{}]",
                    source.kind(),
                    source.display_path(),
                    cols.join(", ")
                )?,
            }
            if !pushdown_predicates.is_empty() {
                let preds: Vec<String> =
                    pushdown_predicates.iter().map(|p| p.display_name()).collect();
                write!(out, ", pushdown=[{}]", preds.join(", "))?;
            }
            if let Some(n) = limit {
                write!(out, ", limit={n}")?;
            }
            writeln!(out)?;
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
        LogicalPlan::Join { left, right, on, filter, join_type, .. } => {
            let on_str: Vec<String> = on
                .iter()
                .map(|(l, r)| format!("{} = {}", l.display_name(), r.display_name()))
                .collect();
            if on_str.is_empty() {
                writeln!(out, "{pad}{}Join", join_type.name())?;
            } else {
                writeln!(out, "{pad}{}Join: on=[{}]", join_type.name(), on_str.join(", "))?;
            }
            if let Some(f) = filter {
                let inner_pad = "  ".repeat(depth + 1);
                writeln!(out, "{inner_pad}filter: {}", f.display_name())?;
            }
            // Two-arm subtree: label each side so reading the tree top-to-bottom
            // still works without ambiguity once joins of joins exist.
            let inner_pad = "  ".repeat(depth + 1);
            writeln!(out, "{inner_pad}L:")?;
            write_indented(left, out, depth + 2)?;
            writeln!(out, "{inner_pad}R:")?;
            write_indented(right, out, depth + 2)?;
        }
    }
    Ok(())
}

impl fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.explain())
    }
}

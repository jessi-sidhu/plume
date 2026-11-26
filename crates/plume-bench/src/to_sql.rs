//! Render a Plume `LogicalPlan` as equivalent DuckDB SQL.
//!
//! Every plan node becomes one subquery, and every output column is
//! aliased to the name in the node's Plume schema, so the SQL produces the
//! same column names (including join collisions like `id`, `id_1`) as the
//! plan. Anything where DuckDB's defaults differ from Plume's is spelled
//! out:
//!
//! - every ORDER BY key says ASC/DESC and NULLS FIRST/LAST;
//! - literals carry their type (`CAST(1 AS INTEGER)`), so DuckDB doesn't
//!   widen or narrow them;
//! - integer division is `//` (DuckDB's `/` on integers returns DOUBLE);
//! - computed and aggregate columns are cast to Plume's result type
//!   (DuckDB's `SUM` of BIGINT is HUGEINT, Plume's is BIGINT).
//!
//! Scans read the file directly (`read_parquet` / `read_csv`), so the SQL
//! needs no views registered. Scan hints the optimizer adds (pushdown
//! predicates, `limit`, `limit_hint`) are ignored: they never change the
//! result, because the Filter and Limit above them stay in the plan.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use arrow_schema::{DataType, Schema, TimeUnit};
use plume_core::{
    AggregateFunction, BinaryOperator, Expr, JoinType, LogicalPlan, ScalarValue, ScanSource,
    SortExpr,
};

/// The DuckDB SQL for `plan`.
pub fn to_sql(plan: &LogicalPlan) -> Result<String> {
    Renderer::default().plan(plan)
}

#[derive(Default)]
struct Renderer {
    next_alias: usize,
}

/// Column name -> SQL for referencing it.
type Resolve<'a> = &'a dyn Fn(&str) -> Result<String>;

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn string_lit(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

fn plain(name: &str) -> Result<String> {
    Ok(quote(name))
}

/// The DuckDB type name for an Arrow type Plume supports.
pub fn sql_type(ty: &DataType) -> Result<&'static str> {
    Ok(match ty {
        DataType::Boolean => "BOOLEAN",
        DataType::Int32 => "INTEGER",
        DataType::Int64 => "BIGINT",
        DataType::Float64 => "DOUBLE",
        DataType::Utf8 => "VARCHAR",
        DataType::Date32 => "DATE",
        DataType::Timestamp(TimeUnit::Microsecond, None) => "TIMESTAMP",
        other => bail!("to_sql: no DuckDB type for {other:?}"),
    })
}

fn path_lit(path: &Path) -> Result<String> {
    let p = path
        .to_str()
        .ok_or_else(|| anyhow!("to_sql: non-UTF-8 path {}", path.display()))?;
    Ok(string_lit(p))
}

impl Renderer {
    fn alias(&mut self, prefix: &str) -> String {
        self.next_alias += 1;
        format!("{prefix}{}", self.next_alias)
    }

    fn plan(&mut self, plan: &LogicalPlan) -> Result<String> {
        match plan {
            LogicalPlan::Scan { source, schema, .. } => {
                let cols: Vec<String> = schema
                    .fields()
                    .iter()
                    .map(|f| format!("{} AS {}", quote(f.name()), quote(f.name())))
                    .collect();
                let from = match source {
                    ScanSource::Parquet { path } => format!("read_parquet({})", path_lit(path)?),
                    ScanSource::Csv { path } => {
                        // Plume reads CSV with the caller's schema and a
                        // header row; tell DuckDB the same instead of
                        // letting it sniff types.
                        let types = schema
                            .fields()
                            .iter()
                            .map(|f| {
                                Ok(format!(
                                    "{}: '{}'",
                                    string_lit(f.name()),
                                    sql_type(f.data_type())?
                                ))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        format!(
                            "read_csv({}, header = true, auto_detect = false, columns = {{{}}})",
                            path_lit(path)?,
                            types.join(", ")
                        )
                    }
                };
                Ok(format!("SELECT {} FROM {from}", cols.join(", ")))
            }
            LogicalPlan::Filter {
                input, predicate, ..
            } => {
                let inner = self.plan(input)?;
                let q = self.alias("q");
                let pred = self.expr(predicate, &input.schema(), &plain)?;
                Ok(format!(
                    "SELECT {} FROM ({inner}) AS {q} WHERE {pred}",
                    pass_through(&input.schema())
                ))
            }
            LogicalPlan::Projection {
                input,
                exprs,
                schema,
            } => {
                let inner = self.plan(input)?;
                let q = self.alias("q");
                let sels = self.select_list(exprs, &input.schema(), schema)?;
                Ok(format!("SELECT {sels} FROM ({inner}) AS {q}"))
            }
            LogicalPlan::Aggregate {
                input,
                group_exprs,
                agg_exprs,
                schema,
            } => {
                let inner = self.plan(input)?;
                let q = self.alias("q");
                let all: Vec<Expr> = group_exprs.iter().chain(agg_exprs).cloned().collect();
                let sels = self.select_list(&all, &input.schema(), schema)?;
                let group = if group_exprs.is_empty() {
                    String::new()
                } else {
                    let ords: Vec<String> =
                        (1..=group_exprs.len()).map(|i| i.to_string()).collect();
                    format!(" GROUP BY {}", ords.join(", "))
                };
                Ok(format!("SELECT {sels} FROM ({inner}) AS {q}{group}"))
            }
            LogicalPlan::Sort { input, exprs, .. } => self.ordered(input, exprs, None),
            LogicalPlan::TopK {
                input, exprs, k, ..
            } => self.ordered(input, exprs, Some(*k)),
            LogicalPlan::Limit { input, n, .. } => {
                // LIMIT directly over a sort goes in the same SELECT, so
                // the order it limits is the sorted one.
                if let LogicalPlan::Sort {
                    input: sort_input,
                    exprs,
                    ..
                } = input.as_ref()
                {
                    return self.ordered(sort_input, exprs, Some(*n));
                }
                let inner = self.plan(input)?;
                let q = self.alias("q");
                Ok(format!(
                    "SELECT {} FROM ({inner}) AS {q} LIMIT {n}",
                    pass_through(&input.schema())
                ))
            }
            LogicalPlan::Join {
                left,
                right,
                on,
                filter,
                join_type,
                schema,
            } => self.join(left, right, on, filter.as_ref(), *join_type, schema),
        }
    }

    /// `SELECT ... ORDER BY keys [LIMIT k]` over `input`.
    fn ordered(
        &mut self,
        input: &LogicalPlan,
        keys: &[SortExpr],
        limit: Option<usize>,
    ) -> Result<String> {
        let inner = self.plan(input)?;
        let q = self.alias("q");
        let schema = input.schema();
        let keys = keys
            .iter()
            .map(|k| {
                Ok(format!(
                    "{} {} NULLS {}",
                    self.expr(&k.expr, &schema, &plain)?,
                    if k.asc { "ASC" } else { "DESC" },
                    if k.nulls_first { "FIRST" } else { "LAST" }
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let limit = limit.map(|n| format!(" LIMIT {n}")).unwrap_or_default();
        Ok(format!(
            "SELECT {} FROM ({inner}) AS {q} ORDER BY {}{limit}",
            pass_through(&schema),
            keys.join(", ")
        ))
    }

    fn join(
        &mut self,
        left: &LogicalPlan,
        right: &LogicalPlan,
        on: &[(Expr, Expr)],
        filter: Option<&Expr>,
        join_type: JoinType,
        schema: &Schema,
    ) -> Result<String> {
        let left_sql = self.plan(left)?;
        let right_sql = self.plan(right)?;
        let (l, r) = (self.alias("l"), self.alias("r"));
        let (ls, rs) = (left.schema(), right.schema());

        // The output schema is the left fields then (unless Semi/Anti) the
        // right fields, renamed on collision. Map each output name back to
        // the side and column it came from.
        let mut sources: Vec<(String, String)> = ls
            .fields()
            .iter()
            .map(|f| (l.clone(), f.name().clone()))
            .collect();
        if !join_type.projects_only_left() {
            sources.extend(rs.fields().iter().map(|f| (r.clone(), f.name().clone())));
        }
        if sources.len() != schema.fields().len() {
            bail!("to_sql: join schema doesn't line up with its inputs");
        }
        let outputs: Vec<(String, String)> = schema
            .fields()
            .iter()
            .zip(&sources)
            .map(|(f, (side, name))| (f.name().clone(), format!("{side}.{}", quote(name))))
            .collect();
        let sels: Vec<String> = outputs
            .iter()
            .map(|(name, src)| format!("{src} AS {}", quote(name)))
            .collect();

        let left_col = |n: &str| Ok(format!("{l}.{}", quote(n)));
        let right_col = |n: &str| Ok(format!("{r}.{}", quote(n)));
        let mut conds = on
            .iter()
            .map(|(a, b)| {
                Ok(format!(
                    "{} = {}",
                    self.expr(a, &ls, &left_col)?,
                    self.expr(b, &rs, &right_col)?
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let output_col = |n: &str| {
            outputs
                .iter()
                .find(|(name, _)| name == n)
                .map(|(_, src)| src.clone())
                .ok_or_else(|| anyhow!("to_sql: unknown join column {n}"))
        };
        let residual = filter
            .map(|f| self.expr(f, schema, &output_col))
            .transpose()?;

        let mut where_clause = String::new();
        let kw = match join_type {
            JoinType::Inner => "INNER JOIN",
            JoinType::Left => "LEFT JOIN",
            JoinType::Right => "RIGHT JOIN",
            JoinType::Full => "FULL JOIN",
            JoinType::Semi => "SEMI JOIN",
            JoinType::Anti => "ANTI JOIN",
            JoinType::Cross => "CROSS JOIN",
        };
        match (join_type, residual) {
            (_, None) => {}
            (JoinType::Inner, Some(f)) => conds.push(f),
            (JoinType::Cross, Some(f)) => where_clause = format!(" WHERE {f}"),
            (jt, Some(_)) => bail!("to_sql: residual filter on a {} join", jt.name()),
        }
        let on_clause = if join_type == JoinType::Cross {
            if !conds.is_empty() {
                bail!("to_sql: cross join with keys");
            }
            String::new()
        } else {
            format!(" ON {}", conds.join(" AND "))
        };
        Ok(format!(
            "SELECT {} FROM ({left_sql}) AS {l} {kw} ({right_sql}) AS {r}{on_clause}{where_clause}",
            sels.join(", ")
        ))
    }

    /// `expr AS name` for each output column. Anything but a bare column is
    /// cast to the type in Plume's output schema.
    fn select_list(&mut self, exprs: &[Expr], input: &Schema, output: &Schema) -> Result<String> {
        if exprs.len() != output.fields().len() {
            bail!("to_sql: select list doesn't match its schema");
        }
        let sels = exprs
            .iter()
            .zip(output.fields())
            .map(|(e, f)| {
                let bare = strip_alias(e);
                let sql = self.expr(bare, input, &plain)?;
                let sql = match (bare, f.data_type()) {
                    (Expr::Column(_), _) | (_, DataType::Null) => sql,
                    (_, ty) => format!("CAST({sql} AS {})", sql_type(ty)?),
                };
                Ok(format!("{sql} AS {}", quote(f.name())))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(sels.join(", "))
    }

    fn expr(&self, e: &Expr, schema: &Schema, col: Resolve) -> Result<String> {
        Ok(match e {
            Expr::Column(name) => col(name)?,
            Expr::Literal(v) => literal(v)?,
            Expr::Alias { expr, .. } => self.expr(expr, schema, col)?,
            Expr::BinaryOp { left, op, right } => {
                let (ls, rs) = (
                    self.expr(left, schema, col)?,
                    self.expr(right, schema, col)?,
                );
                let sym = match op {
                    BinaryOperator::Eq => "=",
                    BinaryOperator::NotEq => "<>",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::LtEq => "<=",
                    BinaryOperator::Gt => ">",
                    BinaryOperator::GtEq => ">=",
                    BinaryOperator::And => "AND",
                    BinaryOperator::Or => "OR",
                    BinaryOperator::Plus => "+",
                    BinaryOperator::Minus => "-",
                    BinaryOperator::Multiply => "*",
                    BinaryOperator::Divide => {
                        let ty = left.data_type(schema)?;
                        if matches!(ty, DataType::Int32 | DataType::Int64) {
                            "//"
                        } else {
                            "/"
                        }
                    }
                };
                format!("({ls} {sym} {rs})")
            }
            Expr::Aggregate { fun, arg } => {
                if *fun == AggregateFunction::CountStar {
                    "count(*)".to_string()
                } else {
                    let name = match fun {
                        AggregateFunction::Count => "count",
                        AggregateFunction::Sum => "sum",
                        AggregateFunction::Avg => "avg",
                        AggregateFunction::Min => "min",
                        AggregateFunction::Max => "max",
                        AggregateFunction::CountStar => unreachable!(),
                    };
                    format!("{name}({})", self.expr(arg, schema, col)?)
                }
            }
            Expr::Not(inner) => format!("(NOT {})", self.expr(inner, schema, col)?),
            Expr::IsNull(inner) => format!("({} IS NULL)", self.expr(inner, schema, col)?),
            Expr::IsNotNull(inner) => {
                format!("({} IS NOT NULL)", self.expr(inner, schema, col)?)
            }
        })
    }
}

fn strip_alias(e: &Expr) -> &Expr {
    match e {
        Expr::Alias { expr, .. } => strip_alias(expr),
        other => other,
    }
}

/// `"a" AS "a", "b" AS "b", ...` for every column of `schema`.
fn pass_through(schema: &Schema) -> String {
    schema
        .fields()
        .iter()
        .map(|f| format!("{} AS {}", quote(f.name()), quote(f.name())))
        .collect::<Vec<_>>()
        .join(", ")
}

fn literal(v: &ScalarValue) -> Result<String> {
    Ok(match v {
        ScalarValue::Null => "NULL".to_string(),
        ScalarValue::Boolean(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        ScalarValue::Int32(i) => format!("CAST({i} AS INTEGER)"),
        ScalarValue::Int64(i) => format!("CAST({i} AS BIGINT)"),
        ScalarValue::Float64(f) if f.is_finite() => format!("CAST({f:?} AS DOUBLE)"),
        ScalarValue::Float64(f) => format!("CAST('{f}' AS DOUBLE)"),
        ScalarValue::Utf8(s) => string_lit(s),
        // display_form already gives DATE '...' / TIMESTAMP '...'.
        ScalarValue::Date32(_) | ScalarValue::TimestampMicros(_) => {
            let s = v.display_form();
            if !s.contains('\'') {
                bail!("to_sql: literal out of range: {s}");
            }
            s
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::Field;
    use plume_core::{col, count_star, lit, not, sum, DataFrame};

    use super::*;

    fn people() -> DataFrame {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("age", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        DataFrame::scan_parquet("/data/people.parquet", schema)
    }

    #[test]
    fn filter_and_project_alias_every_column() {
        let plan = people()
            .filter(col("age").gt(lit(30_i32)).and(not(col("name").is_null())))
            .unwrap()
            .project(vec![col("id"), col("age").div(lit(2_i32)).alias("half")])
            .unwrap()
            .into_plan();
        assert_eq!(
            to_sql(&plan).unwrap(),
            "SELECT \"id\" AS \"id\", CAST((\"age\" // CAST(2 AS INTEGER)) AS INTEGER) AS \"half\" \
             FROM (SELECT \"id\" AS \"id\", \"age\" AS \"age\", \"name\" AS \"name\" \
             FROM (SELECT \"id\" AS \"id\", \"age\" AS \"age\", \"name\" AS \"name\" \
             FROM read_parquet('/data/people.parquet')) AS q1 \
             WHERE ((\"age\" > CAST(30 AS INTEGER)) AND (NOT (\"name\" IS NULL)))) AS q2"
        );
    }

    #[test]
    fn sums_are_cast_and_groups_use_ordinals() {
        let plan = people()
            .aggregate(
                vec![col("name")],
                vec![sum(col("age")).alias("s"), count_star().alias("n")],
            )
            .unwrap()
            .into_plan();
        let sql = to_sql(&plan).unwrap();
        assert!(
            sql.contains("CAST(sum(\"age\") AS BIGINT) AS \"s\""),
            "{sql}"
        );
        assert!(sql.contains("CAST(count(*) AS BIGINT) AS \"n\""), "{sql}");
        assert!(sql.ends_with("GROUP BY 1"), "{sql}");
    }

    #[test]
    fn limit_over_sort_shares_one_select() {
        let plan = people()
            .sort(vec![SortExpr::new(col("age"), false, false)])
            .unwrap()
            .limit(3)
            .into_plan();
        let sql = to_sql(&plan).unwrap();
        assert!(
            sql.ends_with("ORDER BY \"age\" DESC NULLS LAST LIMIT 3"),
            "{sql}"
        );
        assert_eq!(sql.matches("SELECT").count(), 2, "{sql}");
    }

    #[test]
    fn join_collisions_are_aliased() {
        let plan = people()
            .join(people(), vec![(col("id"), col("id"))], JoinType::Left)
            .unwrap()
            .into_plan();
        let sql = to_sql(&plan).unwrap();
        assert!(sql.contains("l1.\"id\" AS \"id\""), "{sql}");
        assert!(sql.contains("r2.\"id\" AS \"id_1\""), "{sql}");
        assert!(sql.contains("LEFT JOIN"), "{sql}");
        assert!(sql.contains("ON l1.\"id\" = r2.\"id\""), "{sql}");
    }

    #[test]
    fn literals_carry_their_types() {
        assert_eq!(
            literal(&ScalarValue::Int64(-5)).unwrap(),
            "CAST(-5 AS BIGINT)"
        );
        assert_eq!(
            literal(&ScalarValue::Float64(2.0)).unwrap(),
            "CAST(2.0 AS DOUBLE)"
        );
        assert_eq!(
            literal(&ScalarValue::Float64(f64::NEG_INFINITY)).unwrap(),
            "CAST('-inf' AS DOUBLE)"
        );
        assert_eq!(
            literal(&ScalarValue::Utf8("it's".into())).unwrap(),
            "'it''s'"
        );
        assert_eq!(
            literal(&ScalarValue::Date32(0)).unwrap(),
            "DATE '1970-01-01'"
        );
    }
}

//! Randomized differential testing: generate a random query, build it both
//! as a Plume `DataFrame` and as DuckDB SQL, run both, compare.
//!
//! Everything comes from one `u64` seed: the data (two small Parquet
//! tables, written fresh per case) and the query. A failure prints the
//! seed, so `cargo run -p plume-bench --bin fuzz -- --seed <s>` replays it.
//!
//! # Data
//!
//! `t1` (0 to ~3,000 rows) and `t2` (0 to ~200 rows) with NULLs in every
//! column but `a_id`, duplicate keys, empty strings, and a row-group size
//! picked per case (7 to 500 rows), so pruning has row groups to skip.
//! `a_id` is sorted and `a_d` / `a_t` increase with it, so range filters
//! and top-k actually prune. Column names are prefixed (`a_` for t1, `b_`
//! for t2) so a join never has to rename anything.
//!
//! Floats are multiples of 0.25 in a small range: sums are exact in both
//! engines, whatever order they add in. The data never holds -0.0 or NaN,
//! which the two engines order differently.
//!
//! # Queries
//!
//! A body, then optionally ORDER BY and LIMIT at the top:
//!
//! - the body starts from a scan of t1, or t1 joined with t2 on
//!   `a_k = b_k` (Inner, Left, Right, Full, Semi or Anti), each side
//!   optionally filtered;
//! - (15% of cases instead take a shape LIMIT and top-k pruning need, see
//!   `pruning_shaped`)
//! - then up to three of: Filter, Projection (columns plus computed
//!   expressions), Aggregate (at most one, 0 to 2 group keys), and a
//!   nested "top n" (ORDER BY every column, LIMIT n), whose result is a
//!   well-defined multiset because ties are whole equal rows.
//!
//! Expressions are well typed by construction (Plume never widens types
//! implicitly): comparisons, AND/OR, and +, -, * on matching numeric
//! types. Integer division is left out (DuckDB's `/` returns a double),
//! values stay small so nothing overflows, and floats are only multiplied
//! by positive constants, so no -0.0 appears.
//!
//! # Checking
//!
//! Each query runs unoptimized, optimized, and optimized with the runtime
//! pruning (top-k, join) switched off. Every run is compared with DuckDB:
//! exactly (any row order) when the result is unique; with the Phase 2
//! checks when it has ORDER BY (sort keys row by row) or LIMIT without
//! ORDER BY (row count plus containment).
//!
//! # Shrinking
//!
//! A failing seed usually comes with thousands of rows and a few operators
//! that have nothing to do with the bug. `shrink` cuts it down: it drops
//! query steps (the join, a filter, an operator, the top-level ORDER BY or
//! LIMIT) and then rows of each table, keeping every cut after which the
//! case still fails the same way (a wrong result, or an error). Which
//! mode fails isn't part of that: with a LIMIT over a hash aggregate, the
//! rows a run returns depend on hash order, so a bug can show up in a
//! different mode from one run to the next. A
//! dropped step still draws its random numbers, so the rest of the query
//! is generated as before; only its output is thrown away. Dropping rows
//! leaves gaps in `a_id`, which stays sorted.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow_array::{
    ArrayRef, Date32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_core::{
    avg, col, count, count_star, lit, lit_date, lit_timestamp, max, min, sum, DataFrame, Expr,
    JoinType, SortExpr,
};
use plume_exec::{collect_with_options, PlannerOptions};
use plume_io::scan_parquet_auto;
use plume_optimizer::Optimizer;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::harness::duckdb_query;
use crate::workloads::Check;

/// Column types the generator uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    I64,
    I32,
    F64,
    Str,
    Date,
    Ts,
}

impl Ty {
    fn numeric(self) -> bool {
        matches!(self, Ty::I64 | Ty::I32 | Ty::F64)
    }
}

#[derive(Debug, Clone)]
struct Col {
    name: String,
    ty: Ty,
}

/// A query under construction: the Plume side, the SQL side, and the
/// output columns they share.
#[derive(Clone)]
struct Q {
    df: DataFrame,
    sql: String,
    cols: Vec<Col>,
}

/// The case's tables: (name, Parquet file).
type Tables = Vec<(&'static str, PathBuf)>;

/// A part of the generated query that shrinking can drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Step {
    /// The join (the body becomes the, maybe filtered, scan of t1).
    Join,
    /// The filter on t1 under the join.
    LeftFilter,
    /// The filter on t2 under the join.
    RightFilter,
    /// The i-th operator over the body.
    Op(usize),
    /// The range filter of a pruning-shaped case.
    RangeFilter,
    /// The top-level ORDER BY.
    OrderBy,
    /// The top-level LIMIT.
    Limit,
}

/// How to cut a seed's case down. The default cuts nothing.
#[derive(Debug, Clone, Default)]
pub struct Shrink {
    pub skip: BTreeSet<Step>,
    /// Rows of t1 to keep, as indexes into the full table; `None` keeps all.
    pub t1_rows: Option<Vec<usize>>,
    /// Same for t2.
    pub t2_rows: Option<Vec<usize>>,
}

/// One generated test case.
pub struct Case {
    pub seed: u64,
    /// Keeps the Parquet files alive.
    _dir: tempfile::TempDir,
    tables: Tables,
    pub df: DataFrame,
    pub sql: String,
    pub check: Check,
    /// The droppable steps this case applied.
    pub steps: Vec<Step>,
    /// Row counts of t1 and t2 before any rows were dropped.
    pub full_rows: (usize, usize),
}

const WORDS: [&str; 7] = ["", "a", "ab", "b", "mm", "z", "Z"];
const DAY0: i32 = 19_000; // 2022-01-08
const US_PER_DAY: i64 = 86_400_000_000;

struct Gen {
    rng: StdRng,
    next_alias: usize,
    shrink: Shrink,
    steps: Vec<Step>,
}

impl Gen {
    /// Whether to apply `step`; records it when applied.
    fn apply(&mut self, step: Step) -> bool {
        if self.shrink.skip.contains(&step) {
            false
        } else {
            self.steps.push(step);
            true
        }
    }
    fn chance(&mut self, p: f64) -> bool {
        self.rng.gen_bool(p)
    }
    fn alias(&mut self, prefix: &str) -> String {
        self.next_alias += 1;
        format!("{prefix}{}", self.next_alias)
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.rng.gen_range(0..xs.len())]
    }
}

// ---- data ---------------------------------------------------------------

fn opt<T>(g: &mut Gen, null_rate: f64, v: T) -> Option<T> {
    if g.chance(null_rate) {
        None
    } else {
        Some(v)
    }
}

/// The rows of `v` listed in `rows`, or all of them.
fn subset<T: Clone>(v: Vec<T>, rows: &Option<Vec<usize>>) -> Vec<T> {
    match rows {
        None => v,
        Some(rows) => rows.iter().map(|&i| v[i].clone()).collect(),
    }
}

fn write(path: &Path, fields: Vec<Field>, cols: Vec<ArrayRef>, rg: usize) -> Result<()> {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), cols)?;
    let props = WriterProperties::builder()
        .set_max_row_group_size(rg)
        .build();
    let mut w = ArrowWriter::try_new(File::create(path)?, schema, Some(props))?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

fn table_sizes(g: &mut Gen) -> (usize, usize) {
    let n1 = *g.pick(&[0usize, 1, 5, 40, 300, 1000, 3000]);
    let n2 = *g.pick(&[0usize, 1, 3, 20, 200]);
    (n1, n2)
}

/// Write t1 and t2; returns their paths and full row counts.
fn write_tables(g: &mut Gen, dir: &Path) -> Result<(Tables, (usize, usize))> {
    let (n1, n2) = table_sizes(g);
    let rg = *g.pick(&[7usize, 64, 500]);
    let nr = *g.pick(&[0.0, 0.1, 0.3]);

    let mut a_k = Vec::with_capacity(n1);
    let mut a_i = Vec::with_capacity(n1);
    let mut a_f = Vec::with_capacity(n1);
    let mut a_s = Vec::with_capacity(n1);
    let mut a_d = Vec::with_capacity(n1);
    let mut a_t = Vec::with_capacity(n1);
    for i in 0..n1 {
        let k = g.rng.gen_range(0..20i64);
        a_k.push(opt(g, nr, k));
        let v = g.rng.gen_range(-50..=50i32);
        a_i.push(opt(g, nr, v));
        let f = g.rng.gen_range(-400..=400i32) as f64 * 0.25;
        a_f.push(opt(g, nr, f));
        let w = *g.pick(&WORDS);
        a_s.push(opt(g, nr, w));
        // Dates and timestamps follow a_id, with some jitter.
        let d = DAY0 + (i / 10) as i32 + g.rng.gen_range(0..3);
        a_d.push(opt(g, nr, d));
        let t =
            (DAY0 as i64) * US_PER_DAY + i as i64 * 3_600_000_000 + g.rng.gen_range(0..1_000_000);
        a_t.push(opt(g, nr, t));
    }
    let r1 = g.shrink.t1_rows.clone();
    let a_id = subset((0..n1 as i64).collect(), &r1);
    let (a_k, a_i, a_f) = (subset(a_k, &r1), subset(a_i, &r1), subset(a_f, &r1));
    let (a_s, a_d, a_t) = (subset(a_s, &r1), subset(a_d, &r1), subset(a_t, &r1));
    let t1 = dir.join("t1.parquet");
    write(
        &t1,
        vec![
            Field::new("a_id", DataType::Int64, false),
            Field::new("a_k", DataType::Int64, true),
            Field::new("a_i", DataType::Int32, true),
            Field::new("a_f", DataType::Float64, true),
            Field::new("a_s", DataType::Utf8, true),
            Field::new("a_d", DataType::Date32, true),
            Field::new(
                "a_t",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
        ],
        vec![
            Arc::new(Int64Array::from(a_id)),
            Arc::new(Int64Array::from(a_k)),
            Arc::new(Int32Array::from(a_i)),
            Arc::new(Float64Array::from(a_f)),
            Arc::new(StringArray::from(a_s)),
            Arc::new(Date32Array::from(a_d)),
            Arc::new(TimestampMicrosecondArray::from(a_t)),
        ],
        rg,
    )?;

    let mut b_k = Vec::with_capacity(n2);
    let mut b_v = Vec::with_capacity(n2);
    let mut b_s = Vec::with_capacity(n2);
    let mut b_f = Vec::with_capacity(n2);
    for _ in 0..n2 {
        let k = g.rng.gen_range(0..25i64);
        b_k.push(opt(g, nr, k));
        let v = g.rng.gen_range(-100..=100i64);
        b_v.push(opt(g, nr, v));
        let w = *g.pick(&WORDS);
        b_s.push(opt(g, nr, w));
        let f = g.rng.gen_range(-40..=40i32) as f64 * 0.25;
        b_f.push(opt(g, nr, f));
    }
    let r2 = g.shrink.t2_rows.clone();
    let (b_k, b_v) = (subset(b_k, &r2), subset(b_v, &r2));
    let (b_s, b_f) = (subset(b_s, &r2), subset(b_f, &r2));
    let t2 = dir.join("t2.parquet");
    write(
        &t2,
        vec![
            Field::new("b_k", DataType::Int64, true),
            Field::new("b_v", DataType::Int64, true),
            Field::new("b_s", DataType::Utf8, true),
            Field::new("b_f", DataType::Float64, true),
        ],
        vec![
            Arc::new(Int64Array::from(b_k)),
            Arc::new(Int64Array::from(b_v)),
            Arc::new(StringArray::from(b_s)),
            Arc::new(Float64Array::from(b_f)),
        ],
        rg,
    )?;
    Ok((vec![("t1", t1), ("t2", t2)], (n1, n2)))
}

fn t1_cols() -> Vec<Col> {
    [
        ("a_id", Ty::I64),
        ("a_k", Ty::I64),
        ("a_i", Ty::I32),
        ("a_f", Ty::F64),
        ("a_s", Ty::Str),
        ("a_d", Ty::Date),
        ("a_t", Ty::Ts),
    ]
    .into_iter()
    .map(|(n, ty)| Col { name: n.into(), ty })
    .collect()
}

fn t2_cols() -> Vec<Col> {
    [
        ("b_k", Ty::I64),
        ("b_v", Ty::I64),
        ("b_s", Ty::Str),
        ("b_f", Ty::F64),
    ]
    .into_iter()
    .map(|(n, ty)| Col { name: n.into(), ty })
    .collect()
}

// ---- expressions --------------------------------------------------------

fn quote(name: &str) -> String {
    format!("\"{name}\"")
}

/// A literal of type `ty`, as (Plume, SQL).
fn literal(g: &mut Gen, ty: Ty) -> (Expr, String) {
    match ty {
        Ty::I64 => {
            let v = g.rng.gen_range(-60..=3100i64);
            (lit(v), format!("CAST({v} AS BIGINT)"))
        }
        Ty::I32 => {
            let v = g.rng.gen_range(-60..=60i32);
            (lit(v), format!("CAST({v} AS INTEGER)"))
        }
        Ty::F64 => {
            let v = g.rng.gen_range(-420..=420i32) as f64 * 0.25;
            (lit(v), format!("CAST({v:?} AS DOUBLE)"))
        }
        Ty::Str => {
            let w = *g.pick(&WORDS);
            (lit(w), format!("'{w}'"))
        }
        Ty::Date => {
            let d = DAY0 + g.rng.gen_range(-5..320);
            let s = date_str(d);
            (lit_date(&s), format!("DATE '{s}'"))
        }
        Ty::Ts => {
            let us = (DAY0 as i64) * US_PER_DAY + g.rng.gen_range(-5..3100i64) * 3_600_000_000;
            let s = ts_str(us);
            (lit_timestamp(&s), format!("TIMESTAMP '{s}'"))
        }
    }
}

fn date_str(days: i32) -> String {
    let d =
        chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap() + chrono::Duration::days(days as i64);
    d.format("%Y-%m-%d").to_string()
}

fn ts_str(us: i64) -> String {
    let t = chrono::DateTime::from_timestamp_micros(us)
        .unwrap()
        .naive_utc();
    t.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// A value expression of type `ty` over `cols`, as (Plume, SQL).
fn value(g: &mut Gen, cols: &[Col], ty: Ty, depth: u32) -> (Expr, String) {
    let of_ty: Vec<&Col> = cols.iter().filter(|c| c.ty == ty).collect();
    let roll = g.rng.gen_range(0..10);
    if ty.numeric() && depth > 0 && roll < 3 {
        let (l, ls) = value(g, cols, ty, depth - 1);
        match (ty, g.rng.gen_range(0..3)) {
            // Floats only get multiplied by a positive constant: no -0.0.
            (Ty::F64, 2) => {
                let c = *g.pick(&[0.5, 1.5, 2.0]);
                (l.mul(lit(c)), format!("({ls} * CAST({c:?} AS DOUBLE))"))
            }
            (_, 2) => {
                // Keep integer products small: column times a small literal.
                let (r, rs) = match ty {
                    Ty::I32 => {
                        let v = g.rng.gen_range(-3..=3i32);
                        (lit(v), format!("CAST({v} AS INTEGER)"))
                    }
                    _ => {
                        let v = g.rng.gen_range(-3..=3i64);
                        (lit(v), format!("CAST({v} AS BIGINT)"))
                    }
                };
                (l.mul(r), format!("({ls} * {rs})"))
            }
            (_, op) => {
                let (r, rs) = value(g, cols, ty, depth - 1);
                if op == 0 {
                    (l.plus(r), format!("({ls} + {rs})"))
                } else {
                    (l.minus(r), format!("({ls} - {rs})"))
                }
            }
        }
    } else if !of_ty.is_empty() && roll < 8 {
        let c = (*g.pick(&of_ty)).clone();
        (col(&c.name), quote(&c.name))
    } else {
        literal(g, ty)
    }
}

/// A boolean predicate over `cols`, as (Plume, SQL).
fn predicate(g: &mut Gen, cols: &[Col], depth: u32) -> (Expr, String) {
    if depth > 0 && g.chance(0.35) {
        let (l, ls) = predicate(g, cols, depth - 1);
        let (r, rs) = predicate(g, cols, depth - 1);
        return if g.chance(0.5) {
            (l.and(r), format!("({ls} AND {rs})"))
        } else {
            (l.or(r), format!("({ls} OR {rs})"))
        };
    }
    // Compare a column (usually) with a value of the same type.
    let c = g.pick(cols).clone();
    let (l, ls) = (col(&c.name), quote(&c.name));
    let (r, rs) = if c.ty.numeric() && g.chance(0.3) {
        value(g, cols, c.ty, 1)
    } else {
        literal(g, c.ty)
    };
    let (op, sym) = *g.pick(&[
        ("eq", "="),
        ("neq", "<>"),
        ("lt", "<"),
        ("lt_eq", "<="),
        ("gt", ">"),
        ("gt_eq", ">="),
    ]);
    let e = match op {
        "eq" => l.eq(r),
        "neq" => l.neq(r),
        "lt" => l.lt(r),
        "lt_eq" => l.lt_eq(r),
        "gt" => l.gt(r),
        _ => l.gt_eq(r),
    };
    // Sometimes put the literal side first, to exercise flipping.
    let sql = if g.chance(0.2) {
        let flipped = match sym {
            "<" => ">",
            "<=" => ">=",
            ">" => "<",
            ">=" => "<=",
            s => s,
        };
        format!("({rs} {flipped} {ls})")
    } else {
        format!("({ls} {sym} {rs})")
    };
    (e, sql)
}

// ---- plan operators -----------------------------------------------------

fn wrap(inner: &str, select: &str, rest: &str, g: &mut Gen) -> String {
    let a = g.alias("q");
    format!("SELECT {select} FROM ({inner}) AS {a}{rest}")
}

fn scan(name: &str, path: &Path, cols: Vec<Col>) -> Result<Q> {
    Ok(Q {
        df: scan_parquet_auto(path)?,
        sql: format!("SELECT * FROM {name}"),
        cols,
    })
}

fn filter(g: &mut Gen, q: Q) -> Result<Q> {
    let (p, ps) = predicate(g, &q.cols, 2);
    let sql = wrap(&q.sql, "*", &format!(" WHERE {ps}"), g);
    Ok(Q {
        df: q.df.filter(p)?,
        sql,
        cols: q.cols,
    })
}

fn project(g: &mut Gen, q: Q) -> Result<Q> {
    let mut exprs = Vec::new();
    let mut sels = Vec::new();
    let mut cols = Vec::new();
    for c in &q.cols {
        if g.chance(0.6) {
            exprs.push(col(&c.name));
            sels.push(quote(&c.name));
            cols.push(c.clone());
        }
    }
    let computed = g.rng.gen_range(0..3);
    let numeric: Vec<Ty> = q
        .cols
        .iter()
        .map(|c| c.ty)
        .filter(|t| t.numeric())
        .collect();
    for _ in 0..computed {
        if numeric.is_empty() {
            break;
        }
        let ty = *g.pick(&numeric);
        let (e, s) = value(g, &q.cols, ty, 2);
        let name = g.alias("e");
        exprs.push(e.alias(&name));
        sels.push(format!("{s} AS {}", quote(&name)));
        cols.push(Col { name, ty });
    }
    if exprs.is_empty() {
        let c = q.cols[0].clone();
        exprs.push(col(&c.name));
        sels.push(quote(&c.name));
        cols.push(c);
    }
    let sql = wrap(&q.sql, &sels.join(", "), "", g);
    Ok(Q {
        df: q.df.project(exprs)?,
        sql,
        cols,
    })
}

fn aggregate(g: &mut Gen, q: Q) -> Result<Q> {
    let n_groups = g.rng.gen_range(0..=2usize).min(q.cols.len());
    let mut group = Vec::new();
    let mut group_sql = Vec::new();
    let mut cols = Vec::new();
    let mut pool = q.cols.clone();
    for _ in 0..n_groups {
        let i = g.rng.gen_range(0..pool.len());
        let c = pool.remove(i);
        group.push(col(&c.name));
        group_sql.push(quote(&c.name));
        cols.push(c);
    }
    let mut aggs = Vec::new();
    let mut agg_sql = Vec::new();
    for _ in 0..g.rng.gen_range(1..=3) {
        let name = g.alias("g");
        let c = g.pick(&q.cols).clone();
        let (e, s, ty) = match g.rng.gen_range(0..6) {
            0 => (count_star(), "count(*)".to_string(), Ty::I64),
            1 => (
                count(col(&c.name)),
                format!("count({})", quote(&c.name)),
                Ty::I64,
            ),
            2 if c.ty.numeric() => {
                let ty = if c.ty == Ty::F64 { Ty::F64 } else { Ty::I64 };
                (sum(col(&c.name)), format!("sum({})", quote(&c.name)), ty)
            }
            3 if c.ty.numeric() => (
                avg(col(&c.name)),
                format!("avg({})", quote(&c.name)),
                Ty::F64,
            ),
            4 => (min(col(&c.name)), format!("min({})", quote(&c.name)), c.ty),
            _ => (max(col(&c.name)), format!("max({})", quote(&c.name)), c.ty),
        };
        aggs.push(e.alias(&name));
        agg_sql.push(format!("{s} AS {}", quote(&name)));
        cols.push(Col { name, ty });
    }
    let mut select = group_sql.clone();
    select.extend(agg_sql);
    let rest = if group_sql.is_empty() {
        String::new()
    } else {
        format!(" GROUP BY {}", group_sql.join(", "))
    };
    let sql = wrap(&q.sql, &select.join(", "), &rest, g);
    Ok(Q {
        df: q.df.aggregate(group, aggs)?,
        sql,
        cols,
    })
}

/// Random direction and NULL placement for each of `cols`.
fn sort_keys(g: &mut Gen, cols: &[Col]) -> (Vec<SortExpr>, String) {
    let mut keys = Vec::new();
    let mut sql = Vec::new();
    for c in cols {
        let asc = g.chance(0.5);
        let nulls_first = g.chance(0.3);
        keys.push(SortExpr::new(col(&c.name), asc, nulls_first));
        sql.push(format!(
            "{} {} NULLS {}",
            quote(&c.name),
            if asc { "ASC" } else { "DESC" },
            if nulls_first { "FIRST" } else { "LAST" }
        ));
    }
    (keys, sql.join(", "))
}

/// ORDER BY every column, LIMIT n: a well-defined multiset, so it can sit
/// anywhere in the plan.
fn top_n(g: &mut Gen, q: Q) -> Result<Q> {
    let mut order = q.cols.clone();
    // Shuffle the key order.
    for i in (1..order.len()).rev() {
        let j = g.rng.gen_range(0..=i);
        order.swap(i, j);
    }
    let (keys, keys_sql) = sort_keys(g, &order);
    let n = g.rng.gen_range(0..25usize);
    let sql = wrap(&q.sql, "*", &format!(" ORDER BY {keys_sql} LIMIT {n}"), g);
    Ok(Q {
        df: q.df.sort(keys)?.limit(n),
        sql,
        cols: q.cols,
    })
}

fn base(g: &mut Gen, tables: &[(&'static str, PathBuf)]) -> Result<Q> {
    let mut left = scan("t1", &tables[0].1, t1_cols())?;
    if !g.chance(0.4) {
        return Ok(left);
    }
    if g.chance(0.4) {
        let f = filter(g, left.clone())?;
        if g.apply(Step::LeftFilter) {
            left = f;
        }
    }
    let mut right = scan("t2", &tables[1].1, t2_cols())?;
    let right_filtered = if g.chance(0.4) {
        Some(filter(g, right.clone())?)
    } else {
        None
    };
    let (jt, kw) = *g.pick(&[
        (JoinType::Inner, "INNER JOIN"),
        (JoinType::Left, "LEFT JOIN"),
        (JoinType::Right, "RIGHT JOIN"),
        (JoinType::Full, "FULL JOIN"),
        (JoinType::Semi, "SEMI JOIN"),
        (JoinType::Anti, "ANTI JOIN"),
    ]);
    let (l, r) = (g.alias("l"), g.alias("r"));
    if !g.apply(Step::Join) {
        return Ok(left);
    }
    if let Some(f) = right_filtered {
        if g.apply(Step::RightFilter) {
            right = f;
        }
    }
    let sql = format!(
        "SELECT * FROM ({}) AS {l} {kw} ({}) AS {r} ON {l}.\"a_k\" = {r}.\"b_k\"",
        left.sql, right.sql
    );
    let mut cols = left.cols;
    if !matches!(jt, JoinType::Semi | JoinType::Anti) {
        cols.extend(right.cols);
    }
    Ok(Q {
        df: left.df.join(right.df, vec![(col("a_k"), col("b_k"))], jt)?,
        sql,
        cols,
    })
}

/// Generate the case for `seed`.
pub fn generate(seed: u64) -> Result<Case> {
    generate_with(seed, &Shrink::default())
}

/// Generate the case for `seed`, cut down by `shrink`.
pub fn generate_with(seed: u64, shrink: &Shrink) -> Result<Case> {
    let mut g = Gen {
        rng: StdRng::seed_from_u64(seed),
        next_alias: 0,
        shrink: shrink.clone(),
        steps: Vec::new(),
    };
    let dir = tempfile::tempdir()?;
    let (tables, full_rows) = write_tables(&mut g, dir.path())?;
    if g.chance(0.15) {
        return pruning_shaped(g, seed, dir, tables, full_rows);
    }
    let mut q = base(&mut g, &tables)?;
    let mut aggregated = false;
    // Keep this draw an i32: another type draws differently and would
    // change every seed's case.
    for i in 0..g.rng.gen_range(0..=3i32) {
        let next = match g.rng.gen_range(0..10) {
            0..=3 => filter(&mut g, q.clone())?,
            4..=6 => project(&mut g, q.clone())?,
            7 | 8 if !aggregated => {
                aggregated = true;
                aggregate(&mut g, q.clone())?
            }
            _ => top_n(&mut g, q.clone())?,
        };
        if g.apply(Step::Op(i as usize)) {
            q = next;
        }
    }

    // Top level: maybe ORDER BY some columns, maybe LIMIT.
    let body_sql = q.sql.clone();
    let sorted = g.chance(0.4);
    let limit = g.chance(0.35).then(|| g.rng.gen_range(0..30usize));
    let limit = limit.filter(|_| g.apply(Step::Limit));
    let order = if sorted {
        let mut pool = q.cols.clone();
        let mut chosen = Vec::new();
        for _ in 0..g.rng.gen_range(1..=pool.len().min(3)) {
            let i = g.rng.gen_range(0..pool.len());
            chosen.push(pool.remove(i));
        }
        let (keys, keys_sql) = sort_keys(&mut g, &chosen);
        g.apply(Step::OrderBy).then_some((chosen, keys, keys_sql))
    } else {
        None
    };
    let (df, sql, check) = if let Some((chosen, keys, keys_sql)) = order {
        let names: Vec<String> = chosen.iter().map(|c| c.name.clone()).collect();
        let mut sql = wrap(&body_sql, "*", &format!(" ORDER BY {keys_sql}"), &mut g);
        let mut df = q.df.sort(keys)?;
        if let Some(n) = limit {
            write!(sql, " LIMIT {n}")?;
            df = df.limit(n);
        }
        (
            df,
            sql,
            Check::TopK {
                keys: names,
                unlimited_sql: body_sql,
            },
        )
    } else if let Some(n) = limit {
        (
            q.df.limit(n),
            wrap(&body_sql, "*", &format!(" LIMIT {n}"), &mut g),
            Check::Limit {
                k: n,
                unlimited_sql: body_sql,
            },
        )
    } else {
        (q.df, body_sql, Check::Exact)
    };
    Ok(Case {
        seed,
        _dir: dir,
        tables,
        df,
        sql,
        check,
        steps: g.steps,
        full_rows,
    })
}

/// A shape random generation rarely hits but LIMIT and top-k pruning need:
/// t1, a pushable range on a clustered column (a_id, a_d or a_t), then
/// LIMIT, or ORDER BY one of those columns with LIMIT.
fn pruning_shaped(
    mut g: Gen,
    seed: u64,
    dir: tempfile::TempDir,
    tables: Tables,
    full_rows: (usize, usize),
) -> Result<Case> {
    let cols = t1_cols();
    let clustered: Vec<Col> = cols
        .iter()
        .filter(|c| matches!(c.name.as_str(), "a_id" | "a_d" | "a_t"))
        .cloned()
        .collect();
    let mut q = scan("t1", &tables[0].1, cols)?;
    let top_k = g.chance(0.5);
    if g.chance(0.8) {
        // LIMIT pruning needs FullMatch groups, which the NULL-free a_id
        // gives most reliably, so the LIMIT branch mostly filters on it.
        let c = if !top_k && g.chance(0.7) {
            clustered[0].clone()
        } else {
            g.pick(&clustered).clone()
        };
        let ((lo, los), (hi, his)) = if c.name == "a_id" {
            // An ordered range on the NULL-free sorted column, so whole row
            // groups fall inside it (FullMatch) and LIMIT pruning can fire.
            let lo = g.rng.gen_range(0..2000i64);
            let hi = lo + g.rng.gen_range(1..1500i64);
            (
                (lit(lo), format!("CAST({lo} AS BIGINT)")),
                (lit(hi), format!("CAST({hi} AS BIGINT)")),
            )
        } else {
            (literal(&mut g, c.ty), literal(&mut g, c.ty))
        };
        let n = quote(&c.name);
        let sql = wrap(
            &q.sql,
            "*",
            &format!(" WHERE {n} >= {los} AND {n} < {his}"),
            &mut g,
        );
        let filtered = Q {
            df: q
                .df
                .clone()
                .filter(col(&c.name).gt_eq(lo).and(col(&c.name).lt(hi)))?,
            sql,
            cols: q.cols.clone(),
        };
        if g.apply(Step::RangeFilter) {
            q = filtered;
        }
    }
    let k = g.rng.gen_range(0..200usize);
    let body_sql = q.sql.clone();
    let order = if top_k {
        let c = g.pick(&clustered).clone();
        let (keys, keys_sql) = sort_keys(&mut g, std::slice::from_ref(&c));
        g.apply(Step::OrderBy).then_some((c, keys, keys_sql))
    } else {
        None
    };
    let (df, sql, check) = if let Some((c, keys, keys_sql)) = order {
        let sql = wrap(
            &body_sql,
            "*",
            &format!(" ORDER BY {keys_sql} LIMIT {k}"),
            &mut g,
        );
        (
            q.df.sort(keys)?.limit(k),
            sql,
            Check::TopK {
                keys: vec![c.name],
                unlimited_sql: body_sql,
            },
        )
    } else {
        let sql = wrap(&body_sql, "*", &format!(" LIMIT {k}"), &mut g);
        (
            q.df.limit(k),
            sql,
            Check::Limit {
                k,
                unlimited_sql: body_sql,
            },
        )
    };
    Ok(Case {
        seed,
        _dir: dir,
        tables,
        df,
        sql,
        check,
        steps: g.steps,
        full_rows,
    })
}

/// The ways each case is run; all must agree with DuckDB.
fn modes() -> Vec<(&'static str, bool, PlannerOptions)> {
    let off = PlannerOptions {
        top_k_pruning: false,
        join_pruning: false,
        join_key_set: false,
        top_k_stats_init: false,
    };
    vec![
        ("unoptimized", false, PlannerOptions::default()),
        ("optimized", true, PlannerOptions::default()),
        ("optimized, runtime pruning off", true, off),
    ]
}

/// What one case exercised, from its optimized plan and the scan metrics
/// of the run with every technique on.
#[derive(Debug, Default, Clone, Copy)]
pub struct Coverage {
    pub join: bool,
    pub aggregate: bool,
    pub top_k: bool,
    pub sort: bool,
    pub limit: bool,
    pub filter_pruned: bool,
    pub limit_pruned: bool,
    pub top_k_pruned: bool,
    pub join_pruned: bool,
    pub filter_skipped: bool,
}

/// How a mode went wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// Plume returned an error (or panicked into one).
    Error,
    /// Plume returned rows that don't match DuckDB's.
    WrongResult,
}

/// A Plume-side failure of one case.
#[derive(Debug)]
pub struct Failure {
    pub mode: &'static str,
    pub kind: FailureKind,
    /// Names the seed, mode, SQL and plan.
    pub error: anyhow::Error,
}

pub enum Outcome {
    Pass(Coverage),
    Fail(Failure),
}

impl Case {
    /// Run every mode and check each against DuckDB. The error names the
    /// seed, mode, SQL and plan.
    pub fn run(&self) -> Result<Coverage> {
        match self.outcome()? {
            Outcome::Pass(cov) => Ok(cov),
            Outcome::Fail(f) => Err(f.error),
        }
    }

    /// Like `run`, but tells a Plume failure apart from a problem with
    /// the reference side (an `Err`: DuckDB rejected the SQL, say).
    pub fn outcome(&self) -> Result<Outcome> {
        let mut cov = Coverage::default();
        let conn = crate::harness::reference_connection()?;
        for (name, path) in &self.tables {
            conn.execute_batch(&format!(
                "CREATE VIEW {name} AS SELECT * FROM read_parquet('{}');",
                path.display()
            ))?;
        }
        let views: Vec<(&str, &Path)> =
            self.tables.iter().map(|(n, p)| (*n, p.as_path())).collect();
        let expected = duckdb_query(&self.sql, &[], &views)
            .with_context(|| format!("seed {}: DuckDB failed on\n{}", self.seed, self.sql))?;
        let raw = self.df.clone().into_plan();
        let fail = |mode, kind, error| Ok(Outcome::Fail(Failure { mode, kind, error }));
        for (mode, optimize, opts) in modes() {
            let plan = if optimize {
                match Optimizer::default().optimize(raw.clone()) {
                    Ok(p) => p,
                    Err(e) => {
                        let e = anyhow::Error::from(e).context(format!(
                            "seed {} ({mode}): optimizer failed\nSQL: {}",
                            self.seed, self.sql
                        ));
                        return fail(mode, FailureKind::Error, e);
                    }
                }
            } else {
                raw.clone()
            };
            let ctx = || {
                format!(
                    "seed {} ({mode}):\nSQL: {}\nplan:\n{}",
                    self.seed,
                    self.sql,
                    plan.explain()
                )
            };
            let (actual, metrics) = match collect_with_options(&plan, opts).with_context(ctx) {
                Ok(r) => r,
                Err(e) => return fail(mode, FailureKind::Error, e),
            };
            if let Err(e) = self
                .check
                .verify(&actual, &expected, &conn)
                .with_context(ctx)
            {
                return fail(mode, FailureKind::WrongResult, e);
            }
            if optimize && opts == PlannerOptions::default() {
                let text = plan.explain();
                cov.join = text.contains("Join");
                cov.aggregate = text.contains("Aggregate:");
                cov.top_k = text.contains("TopK:");
                cov.sort = text.contains("Sort:");
                cov.limit = text.contains("Limit:");
                for m in &metrics {
                    cov.filter_pruned |= m.row_groups_pruned > 0;
                    cov.limit_pruned |= m.row_groups_pruned_limit > 0;
                    cov.top_k_pruned |= m.row_groups_pruned_topk > 0;
                    cov.join_pruned |= m.row_groups_pruned_join > 0;
                    cov.filter_skipped |= m.rows_filter_skipped > 0;
                }
            }
        }
        Ok(Outcome::Pass(cov))
    }

    /// The case's Parquet files, by table name.
    pub fn tables(&self) -> &[(&'static str, PathBuf)] {
        &self.tables
    }
}

// ---- shrinking ----------------------------------------------------------

/// A failing case, cut down as far as it would go.
pub struct Shrunk {
    pub case: Case,
    pub shrink: Shrink,
    pub failure: Failure,
    /// Candidate cases tried.
    pub attempts: usize,
}

/// Default number of candidates `shrink` tries.
pub const SHRINK_BUDGET: usize = 2000;

/// Cut `seed`'s case down while it keeps failing the same way (same
/// `FailureKind`), trying at most `budget` candidates. `Ok(None)` if it
/// passes.
pub fn shrink(seed: u64, budget: usize) -> Result<Option<Shrunk>> {
    let case = generate(seed)?;
    let Outcome::Fail(first) = case.outcome()? else {
        return Ok(None);
    };
    let kind = first.kind;
    let mut still_fails = |c: &Case| -> Option<Failure> {
        match c.outcome() {
            Ok(Outcome::Fail(f)) if f.kind == kind => Some(f),
            _ => None,
        }
    };
    Ok(Some(shrink_by(case, first, budget, &mut still_fails)))
}

/// The shrinking loop, over any notion of "still fails" (tests use a fake
/// one). `case` must fail with `failure`.
pub fn shrink_by(
    case: Case,
    failure: Failure,
    budget: usize,
    still_fails: &mut dyn FnMut(&Case) -> Option<Failure>,
) -> Shrunk {
    let mut best = Shrunk {
        case,
        shrink: Shrink::default(),
        failure,
        attempts: 0,
    };
    loop {
        let mut progress = false;
        let mut steps = best.case.steps.clone();
        steps.sort();
        for step in steps {
            let mut cand = best.shrink.clone();
            if cand.skip.insert(step) {
                progress |= attempt(&mut best, cand, budget, still_fails);
            }
        }
        for table in [0, 1] {
            progress |= shrink_rows(&mut best, table, budget, still_fails);
        }
        if !progress || best.attempts >= budget {
            return best;
        }
    }
}

/// Try `cand`; keep it if it still fails.
fn attempt(
    best: &mut Shrunk,
    cand: Shrink,
    budget: usize,
    still_fails: &mut dyn FnMut(&Case) -> Option<Failure>,
) -> bool {
    if best.attempts >= budget {
        return false;
    }
    best.attempts += 1;
    let Ok(case) = generate_with(best.case.seed, &cand) else {
        return false;
    };
    match still_fails(&case) {
        Some(failure) => {
            best.case = case;
            best.shrink = cand;
            best.failure = failure;
            true
        }
        None => false,
    }
}

/// Drop rows of t1 (`table` 0) or t2 (1): chunks of half the rows, then
/// quarters, and so on down to single rows.
fn shrink_rows(
    best: &mut Shrunk,
    table: usize,
    budget: usize,
    still_fails: &mut dyn FnMut(&Case) -> Option<Failure>,
) -> bool {
    let rows = |s: &Shrink| if table == 0 { &s.t1_rows } else { &s.t2_rows }.clone();
    let full = if table == 0 {
        best.case.full_rows.0
    } else {
        best.case.full_rows.1
    };
    let mut kept = rows(&best.shrink).unwrap_or_else(|| (0..full).collect());
    let mut progress = false;
    let mut chunk = kept.len().div_ceil(2);
    while chunk > 0 && !kept.is_empty() {
        let mut i = 0;
        while i < kept.len() {
            if best.attempts >= budget {
                return progress;
            }
            let end = (i + chunk).min(kept.len());
            let fewer: Vec<usize> = kept[..i].iter().chain(&kept[end..]).copied().collect();
            let mut cand = best.shrink.clone();
            if table == 0 {
                cand.t1_rows = Some(fewer.clone());
            } else {
                cand.t2_rows = Some(fewer.clone());
            }
            if attempt(best, cand, budget, still_fails) {
                kept = fewer;
                progress = true;
            } else {
                i = end;
            }
        }
        if chunk == 1 {
            break;
        }
        chunk = chunk.div_ceil(2);
    }
    progress
}

impl Shrunk {
    /// A readable account: what was dropped, the SQL, the remaining
    /// rows (when few), and the failure.
    pub fn report(&self) -> String {
        let c = &self.case;
        let kept = |rows: &Option<Vec<usize>>, full: usize| rows.as_ref().map_or(full, Vec::len);
        let (n1, n2) = (
            kept(&self.shrink.t1_rows, c.full_rows.0),
            kept(&self.shrink.t2_rows, c.full_rows.1),
        );
        let mut out = String::new();
        let _ = writeln!(
            out,
            "shrunk seed {} in {} attempts: dropped {:?}; t1 {} of {} rows, t2 {} of {} rows",
            c.seed, self.attempts, self.shrink.skip, n1, c.full_rows.0, n2, c.full_rows.1
        );
        let _ = writeln!(out, "SQL: {}", c.sql);
        for (name, path) in &c.tables {
            match table_rows(path, 20) {
                Ok(Some(rows)) => {
                    let _ = writeln!(out, "{name}:");
                    for r in rows {
                        let _ = writeln!(out, "  {r}");
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    let _ = writeln!(out, "{name}: can't read: {e:#}");
                }
            }
        }
        let _ = write!(
            out,
            "{:?} in {}: {:#}",
            self.failure.kind, self.failure.mode, self.failure.error
        );
        out
    }

    /// Copy the shrunk tables and the SQL into `dir`.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        for (name, path) in &self.case.tables {
            std::fs::copy(path, dir.join(format!("{name}.parquet")))?;
        }
        std::fs::write(dir.join("query.sql"), format!("{}\n", self.case.sql))?;
        Ok(())
    }
}

/// A table's rows as text, or `None` if it has more than `max`.
fn table_rows(path: &Path, max: usize) -> Result<Option<Vec<String>>> {
    let conn = crate::harness::reference_connection()?;
    let p = path.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?;
    let n: usize = conn.query_row(
        &format!("SELECT count(*) FROM read_parquet('{p}')"),
        [],
        |r| r.get(0),
    )?;
    if n > max {
        return Ok(None);
    }
    let mut stmt = conn.prepare(&format!("SELECT * FROM read_parquet('{p}')"))?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let width = row.as_ref().column_count();
        let mut vals = Vec::with_capacity(width);
        for i in 0..width {
            let v: duckdb::types::Value = row.get(i)?;
            vals.push(format!("{v:?}"));
        }
        out.push(vals.join(", "));
    }
    Ok(Some(out))
}

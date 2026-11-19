//! Randomized differential testing on the real datasets (TPC-H lineitem
//! and orders, NYC taxi trips and zones), in the Phase 1 and 2 layouts.
//!
//! The generated tables in `random_plans` are small, so they can't show
//! what millions of rows in real row groups do: skewed values, long
//! strings, outliers, and layouts where pruning skips most of a file. This
//! runs the same generator over the real files:
//!
//! - literals come from the data: a deterministic sample of each column's
//!   distinct values, so filters select something and range filters land
//!   inside the data;
//! - 30% of cases filter the fact table on a range of a column some layout
//!   sorts by (shipdate, orderkey, quantity; pickup time, pickup zone),
//!   which is what lets filter, LIMIT and top-k pruning skip row groups;
//! - the body joins on the real key (`l_orderkey = o_orderkey`,
//!   `PULocationID = LocationID`);
//! - a quarter of cases take the exact shapes the Phase 2 techniques
//!   prune (see `pruning_shaped`), which random generation rarely hits on
//!   tables this wide;
//! - every query ends in a LIMIT, so no result is millions of rows;
//! - SUM and AVG never take a float column: the data's floats aren't exact
//!   in binary, and two engines adding them in different orders can
//!   differ in the last bits;
//! - only the optimized plans run, with and without runtime pruning: an
//!   unoptimized plan reads every column and fully sorts before a LIMIT,
//!   and the generated-table tests cover that mode.
//!
//! The seed picks the layout (`seed % layouts`), so a seed replays exactly.
//! Shrinking drops query steps only: the tables are the real files.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow::array::{Array, AsArray};
use arrow::datatypes::{
    Date32Type, Float64Type, Int32Type, Int64Type, TimeUnit, TimestampMicrosecondType,
};
use arrow_schema::DataType;
use plume_core::{col, lit, lit_date, lit_timestamp, Expr, JoinType};
use plume_io::scan_parquet_auto;
use rand::Rng;

use crate::harness::duckdb_query;
use crate::random_plans::{
    base, date_str, finish, quote, scan, shrink_with, sort_keys, ts_str, wrap, Case, Col, Gen,
    Samples, Shrink, Shrunk, Source, Step, Ty, Q,
};
use crate::workloads::Check;
use crate::workloads::{layouts, Dataset, Tables};

/// Distinct values sampled per column.
const SAMPLES: usize = 48;

/// One dataset's tables in one layout, with the column samples.
pub struct RealSource {
    pub dataset: Dataset,
    pub layout: &'static str,
    left: (&'static str, std::path::PathBuf, Vec<Col>),
    right: (&'static str, std::path::PathBuf, Vec<Col>),
    keys: (&'static str, &'static str),
    /// Fact-table columns a layout sorts by.
    clustered: &'static [&'static str],
    /// Dimension-table columns where `= value` keeps few rows.
    narrow: &'static [&'static str],
    samples: Arc<Samples>,
}

impl RealSource {
    /// Every layout of `dataset` under `data_dir`, in `layouts()` order.
    /// Samples are drawn once (from the first layout; they are the same
    /// values in every layout).
    pub fn load_all(data_dir: &Path, dataset: Dataset) -> Result<Vec<RealSource>> {
        let mut out: Vec<RealSource> = Vec::new();
        for &layout in layouts(dataset) {
            let tables = Tables::new(data_dir, dataset, layout);
            if !tables.exist() {
                bail!(
                    "{} {layout} files missing under {}; run prepare_datasets",
                    dataset.name(),
                    data_dir.display()
                );
            }
            type Names = &'static [&'static str];
            let (names, keys, clustered, narrow): (_, _, Names, Names) = match dataset {
                Dataset::Tpch => (
                    ("lineitem", "orders"),
                    ("l_orderkey", "o_orderkey"),
                    &["l_shipdate", "l_orderkey", "l_quantity"],
                    &["o_custkey", "o_orderdate", "o_clerk", "o_orderkey"],
                ),
                Dataset::Taxi => (
                    ("trips", "zones"),
                    ("PULocationID", "LocationID"),
                    &["tpep_pickup_datetime", "PULocationID"],
                    &["Borough", "Zone", "service_zone", "LocationID"],
                ),
            };
            let table = |name: &'static str| -> Result<_> {
                let path = tables.path(name)?.to_path_buf();
                let cols = columns(&path)?;
                Ok((name, path, cols))
            };
            let (left, right) = (table(names.0)?, table(names.1)?);
            let samples = match out.first() {
                Some(first) => first.samples.clone(),
                None => {
                    let mut s = sample(&left.1, &left.2)?;
                    s.extend(sample(&right.1, &right.2)?);
                    Arc::new(s)
                }
            };
            out.push(RealSource {
                dataset,
                layout,
                left,
                right,
                keys,
                clustered,
                narrow,
                samples,
            });
        }
        Ok(out)
    }
}

/// The generator's view of a file's columns.
fn columns(path: &Path) -> Result<Vec<Col>> {
    let schema = scan_parquet_auto(path)?.schema();
    schema
        .fields()
        .iter()
        .map(|f| {
            let ty = match f.data_type() {
                DataType::Int64 => Ty::I64,
                DataType::Int32 => Ty::I32,
                DataType::Float64 => Ty::F64,
                DataType::Utf8 => Ty::Str,
                DataType::Date32 => Ty::Date,
                DataType::Timestamp(TimeUnit::Microsecond, None) => Ty::Ts,
                other => bail!(
                    "{}: column {} has unsupported type {other}",
                    path.display(),
                    f.name()
                ),
            };
            Ok(Col {
                name: f.name().clone(),
                ty,
            })
        })
        .collect()
}

/// Up to `SAMPLES` distinct non-NULL values of each column, picked by
/// hash so the choice depends only on the values (not the layout or
/// DuckDB's thread scheduling), sorted ascending.
fn sample(path: &Path, cols: &[Col]) -> Result<Samples> {
    let mut out = HashMap::new();
    for c in cols {
        let n = quote(&c.name);
        let sql = format!(
            "SELECT v FROM (SELECT DISTINCT {n} AS v FROM t WHERE {n} IS NOT NULL \
             ORDER BY hash(v), v LIMIT {SAMPLES}) ORDER BY v"
        );
        let batches = duckdb_query(&sql, &[], &[("t", path)])
            .with_context(|| format!("sample {} of {}", c.name, path.display()))?;
        let mut vals = Vec::new();
        for b in &batches {
            let a = b.column(0);
            for i in 0..a.len() {
                vals.push(literal_of(a.as_ref(), i, c.ty)?);
            }
        }
        out.insert(c.name.clone(), vals);
    }
    Ok(out)
}

/// Row `i` of `a` as a (Plume, SQL) literal of type `ty`.
fn literal_of(a: &dyn Array, i: usize, ty: Ty) -> Result<(Expr, String)> {
    Ok(match (ty, a.data_type()) {
        (Ty::I64, DataType::Int64) => {
            let v = a.as_primitive::<Int64Type>().value(i);
            (lit(v), format!("CAST({v} AS BIGINT)"))
        }
        (Ty::I32, DataType::Int32) => {
            let v = a.as_primitive::<Int32Type>().value(i);
            (lit(v), format!("CAST({v} AS INTEGER)"))
        }
        (Ty::F64, DataType::Float64) => {
            let v = a.as_primitive::<Float64Type>().value(i);
            (lit(v), format!("CAST({v:?} AS DOUBLE)"))
        }
        (Ty::Str, DataType::Utf8) => {
            let v = a.as_string::<i32>().value(i);
            (lit(v), format!("'{}'", v.replace('\'', "''")))
        }
        (Ty::Date, DataType::Date32) => {
            let s = date_str(a.as_primitive::<Date32Type>().value(i));
            (lit_date(&s), format!("DATE '{s}'"))
        }
        (Ty::Ts, DataType::Timestamp(TimeUnit::Microsecond, _)) => {
            // To the second: still a value inside the data's range.
            let s = ts_str(a.as_primitive::<TimestampMicrosecondType>().value(i));
            (lit_timestamp(&s), format!("TIMESTAMP '{s}'"))
        }
        (ty, dt) => bail!("sampled a {dt} for a {ty:?} column"),
    })
}

/// The source for `seed`: layouts take turns.
pub fn source_for(sources: &[RealSource], seed: u64) -> &RealSource {
    &sources[(seed % sources.len() as u64) as usize]
}

/// Generate the case for `seed` over `sources` (from `load_all`).
pub fn generate(sources: &[RealSource], seed: u64) -> Result<Case> {
    generate_with(sources, seed, &Shrink::default())
}

/// Generate the case for `seed`, cut down by `shrink`.
pub fn generate_with(sources: &[RealSource], seed: u64, shrink: &Shrink) -> Result<Case> {
    let src = source_for(sources, seed);
    let mut g = Gen::new(seed, shrink);
    g.samples = src.samples.clone();
    g.float_sums = false;
    let source = Source {
        left: src.left.clone(),
        right: src.right.clone(),
        keys: (src.keys.0.into(), src.keys.1.into()),
    };
    let tables = vec![
        (src.left.0, src.left.1.clone()),
        (src.right.0, src.right.1.clone()),
    ];
    let (df, sql, check) = if g.chance(0.25) {
        pruning_shaped(&mut g, src)?
    } else {
        random_body(&mut g, src, &source)?
    };
    Ok(Case {
        seed,
        _dir: tempfile::tempdir()?,
        tables,
        df,
        sql,
        check,
        optimized_only: true,
        steps: g.steps,
        full_rows: (0, 0),
    })
}

/// The general case: a random body over the source, maybe range-filtered.
fn random_body(
    g: &mut Gen,
    src: &RealSource,
    source: &Source,
) -> Result<(plume_core::DataFrame, String, Check)> {
    let mut q = base(g, source)?;
    if g.chance(0.3) {
        let r = range_filter(g, src, q.clone())?;
        if g.apply(Step::RangeFilter) {
            q = r;
        }
    }
    finish(g, q, true)
}

/// The shapes the Phase 2 techniques prune, over the fact table, maybe
/// range-filtered on a clustered column:
///
/// - LIMIT k (LIMIT pruning reads fully matching row groups first);
/// - ORDER BY a clustered column LIMIT k (top-k pruning);
/// - joined with the dimension table filtered to `column = value`, so the
///   build side has few keys (join pruning), as an Inner or Semi join, or
///   a Left join where join pruning must not fire; then random operators.
fn pruning_shaped(g: &mut Gen, src: &RealSource) -> Result<(plume_core::DataFrame, String, Check)> {
    let mut q = scan(src.left.0, &src.left.1, src.left.2.clone())?;
    if g.chance(0.7) {
        let r = range_filter(g, src, q.clone())?;
        if g.apply(Step::RangeFilter) {
            q = r;
        }
    }
    let k = g.rng.gen_range(0..200usize);
    let body_sql = q.sql.clone();
    Ok(match g.rng.gen_range(0..3) {
        0 => (
            q.df.limit(k),
            wrap(&body_sql, "*", &format!(" LIMIT {k}"), g),
            Check::Limit {
                k,
                unlimited_sql: body_sql,
            },
        ),
        1 => {
            let name = *g.pick(src.clustered);
            let c = q.cols.iter().find(|c| c.name == name).unwrap().clone();
            let (keys, keys_sql) = sort_keys(g, std::slice::from_ref(&c));
            let sql = wrap(
                &body_sql,
                "*",
                &format!(" ORDER BY {keys_sql} LIMIT {k}"),
                g,
            );
            (
                q.df.sort(keys)?.limit(k),
                sql,
                Check::TopK {
                    keys: vec![c.name],
                    unlimited_sql: body_sql,
                },
            )
        }
        _ => {
            let right = scan(src.right.0, &src.right.1, src.right.2.clone())?;
            let name = *g.pick(src.narrow);
            let c = right.cols.iter().find(|c| c.name == name).unwrap().clone();
            let vals = &src.samples[&c.name];
            let (v, vs) = g.pick(vals).clone();
            let right = Q {
                df: right.df.filter(col(&c.name).eq(v))?,
                sql: wrap(
                    &right.sql,
                    "*",
                    &format!(" WHERE {} = {vs}", quote(&c.name)),
                    g,
                ),
                cols: right.cols,
            };
            let (jt, kw) = *g.pick(&[
                (JoinType::Inner, "INNER JOIN"),
                (JoinType::Semi, "SEMI JOIN"),
                (JoinType::Left, "LEFT JOIN"),
            ]);
            let (l, r) = (g.alias("l"), g.alias("r"));
            let (lk, rk) = src.keys;
            let sql = format!(
                "SELECT * FROM ({}) AS {l} {kw} ({}) AS {r} ON {l}.{} = {r}.{}",
                q.sql,
                right.sql,
                quote(lk),
                quote(rk)
            );
            let mut cols = q.cols;
            if !matches!(jt, JoinType::Semi) {
                cols.extend(right.cols);
            }
            let joined = Q {
                df: q.df.join(right.df, vec![(col(lk), col(rk))], jt)?,
                sql,
                cols,
            };
            finish(g, joined, true)?
        }
    })
}

/// `lo <= c AND c <= hi` on a clustered column, from two of its samples.
fn range_filter(g: &mut Gen, src: &RealSource, q: Q) -> Result<Q> {
    let present: Vec<Col> = q
        .cols
        .iter()
        .filter(|c| src.clustered.contains(&c.name.as_str()))
        .cloned()
        .collect();
    if present.is_empty() {
        return Ok(q);
    }
    let c = g.pick(&present).clone();
    let vals = &src.samples[&c.name];
    if vals.is_empty() {
        return Ok(q);
    }
    let i = g.rng.gen_range(0..vals.len());
    let j = g.rng.gen_range(i..vals.len());
    let ((lo, los), (hi, his)) = (vals[i].clone(), vals[j].clone());
    let n = quote(&c.name);
    let sql = wrap(
        &q.sql,
        "*",
        &format!(" WHERE {n} >= {los} AND {n} <= {his}"),
        g,
    );
    Ok(Q {
        df: q
            .df
            .filter(col(&c.name).gt_eq(lo).and(col(&c.name).lt_eq(hi)))?,
        sql,
        cols: q.cols,
    })
}

/// Shrink a failing real-data seed (query steps only).
pub fn shrink(sources: &[RealSource], seed: u64, budget: usize) -> Result<Option<Shrunk>> {
    shrink_with(seed, budget, &|s| generate_with(sources, seed, s))
}

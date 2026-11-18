//! Benchmark queries: each defined once, as a Plume `DataFrame` builder
//! plus the equivalent DuckDB SQL. `for_dataset` is the Phase 1 set
//! (filter pruning); `phase2_for_dataset` adds the LIMIT, top-k and join
//! queries of Phase 2. Every query is tagged with the technique it
//! measures and how its result is checked.
//!
//! The data comes from `prepare_datasets` (see `data/manifest.json`). A
//! query is written against logical table names, not files: `lineitem`
//! and `orders` for TPC-H, `trips` for taxi. [`Tables`] maps those names
//! to the Parquet files of one layout, so the same query runs unchanged
//! on the random, sorted and compound layouts.
//!
//! Every non-join query returns a `count(*)` column named `n` so the
//! benchmark runner can report the real selectivity next to the target.
//!
//! # Selectivity sweep
//!
//! Ranges on the layout sort key (`l_shipdate` / `tpep_pickup_datetime`)
//! sized to match roughly 0.1%, 1%, 10% and 50% of rows. The bounds are
//! fixed literals chosen from the data's span, not tuned per layout:
//!
//! - TPC-H `l_shipdate` covers 1992-01-02..1998-12-01 (~2,500 days, close
//!   to uniform), so the ranges are 3, 25, 250 and 1,260 days.
//! - Taxi covers Jan–Jun 2023 (181 days) plus a few outlier timestamps,
//!   which are kept on purpose. The ranges are 4 hours, 2 days, 18 days
//!   and 90 days. Trip volume varies by hour and weekday, so the real
//!   selectivity drifts from the target; the runner records the actual
//!   value from `n`.
//!
//! # Queries Plume can't express yet
//!
//! Left out rather than approximated in a way that would change what they
//! measure:
//! - TPC-H Q1's `ORDER BY` (no sort operator). Q1 here is its
//!   filter-and-aggregate part; results are compared order-insensitively.
//! - Q12's `CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') ...`
//!   high/low buckets (no CASE, no IN). Q12 here groups by
//!   `o_orderpriority` directly and counts, which reads the same rows.
//!   `l_shipmode IN ('MAIL','SHIP')` is written as an OR.
//! - Q14 (`p_type LIKE 'PROMO%'`, needs LIKE and CASE) and Q13
//!   (`NOT LIKE`, LEFT JOIN + COUNT over a subquery).
//! - Top-k queries (`ORDER BY ... LIMIT k`) are in the Phase 2 set.
//! - Taxi queries on hour-of-day or day-of-week (needs EXTRACT/date
//!   functions).
//! - Date arithmetic in predicates (`DATE '1998-12-01' - INTERVAL 90 DAY`):
//!   the bounds are precomputed literals in both engines.

use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use arrow_array::RecordBatch;
use duckdb::Connection;
use plume_core::{avg, col, count_star, lit, lit_date, lit_timestamp, sum, DataFrame, JoinType};
use plume_io::scan_parquet_auto;

use crate::harness::{assert_batches_match, assert_rows_within, assert_sort_keys_match};

/// Which dataset a query reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    Tpch,
    Taxi,
}

impl Dataset {
    pub fn name(self) -> &'static str {
        match self {
            Dataset::Tpch => "tpch",
            Dataset::Taxi => "taxi",
        }
    }
}

/// The three layouts `prepare_datasets` writes for each fact table (the
/// Phase 1 set).
pub const LAYOUTS: [&str; 3] = ["random", "sorted", "compound"];

/// Every layout of `dataset`'s fact table: the Phase 1 three, plus
/// `orderkey` for TPC-H (lineitem sorted by the join key).
pub fn layouts(dataset: Dataset) -> &'static [&'static str] {
    match dataset {
        Dataset::Tpch => &["random", "sorted", "compound", "orderkey"],
        Dataset::Taxi => &LAYOUTS,
    }
}

/// The pruning technique a query is there to measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Technique {
    /// Min/max pruning on filter predicates (Phase 1, §3).
    Filter,
    /// LIMIT pruning (§4).
    Limit,
    /// Top-k pruning (§5).
    TopK,
    /// Join pruning (§6).
    Join,
}

impl Technique {
    pub fn name(self) -> &'static str {
        match self {
            Technique::Filter => "filter",
            Technique::Limit => "limit",
            Technique::TopK => "topk",
            Technique::Join => "join",
        }
    }
}

/// How a query's Plume result is checked against DuckDB.
pub enum Check {
    /// The same rows as DuckDB's result for `sql`, in any order.
    Exact,
    /// LIMIT without ORDER BY: any `k` matching rows are correct. Exactly
    /// min(k, matches) rows, each in `unlimited_sql`'s result.
    Limit { k: usize, unlimited_sql: String },
    /// ORDER BY ... LIMIT: rows tied on the sort keys may differ. The
    /// `keys` columns match DuckDB's ordered result row by row, and every
    /// row is in `unlimited_sql`'s result.
    TopK {
        keys: Vec<String>,
        unlimited_sql: String,
    },
}

impl Check {
    /// Check Plume's `actual` result. `expected` is DuckDB's result for the
    /// query; `conn` has the query's tables registered as views (for
    /// checks whose reference is too large to pull out of DuckDB).
    pub fn verify(
        &self,
        actual: &[RecordBatch],
        expected: &[RecordBatch],
        conn: &Connection,
    ) -> Result<()> {
        match self {
            Check::Exact => assert_batches_match(actual, expected, false),
            Check::Limit { k, unlimited_sql } => {
                let total: i64 = conn.query_row(
                    &format!("SELECT count(*) FROM ({unlimited_sql})"),
                    [],
                    |r| r.get(0),
                )?;
                let got: usize = actual.iter().map(|b| b.num_rows()).sum();
                let want = (*k).min(total as usize);
                ensure!(
                    got == want,
                    "LIMIT {k}: got {got} rows, expected min({k}, {total}) = {want}"
                );
                assert_rows_within(conn, actual, unlimited_sql)
            }
            Check::TopK {
                keys,
                unlimited_sql,
            } => {
                let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
                assert_sort_keys_match(actual, expected, &keys)?;
                assert_rows_within(conn, actual, unlimited_sql)
            }
        }
    }
}

/// Parquet files for one dataset in one layout, keyed by the table names
/// the queries use.
#[derive(Debug, Clone)]
pub struct Tables {
    pub dataset: Dataset,
    pub layout: String,
    tables: Vec<(&'static str, PathBuf)>,
}

impl Tables {
    /// Files under `data_dir` (the workspace `data/`) for `layout`. Only
    /// the fact table changes with the layout; `orders` and `zones` are
    /// written once.
    pub fn new(data_dir: &Path, dataset: Dataset, layout: &str) -> Self {
        let tables = match dataset {
            Dataset::Tpch => vec![
                (
                    "lineitem",
                    data_dir.join(format!("tpch/lineitem_{layout}.parquet")),
                ),
                ("orders", data_dir.join("tpch/orders.parquet")),
            ],
            Dataset::Taxi => vec![
                (
                    "trips",
                    data_dir.join(format!("taxi/trips_{layout}.parquet")),
                ),
                ("zones", data_dir.join("taxi/zones.parquet")),
            ],
        };
        Tables {
            dataset,
            layout: layout.to_string(),
            tables,
        }
    }

    pub fn path(&self, table: &str) -> Result<&Path> {
        self.tables
            .iter()
            .find(|(name, _)| *name == table)
            .map(|(_, p)| p.as_path())
            .with_context(|| format!("no table {table} in {:?}", self.dataset))
    }

    /// Plume scan of `table`, schema read from the file.
    pub fn scan(&self, table: &str) -> Result<DataFrame> {
        scan_parquet_auto(self.path(table)?)
    }

    /// `(name, path)` pairs in the form `harness::duckdb_query` takes.
    pub fn duckdb_views(&self) -> Vec<(&str, &Path)> {
        self.tables.iter().map(|(n, p)| (*n, p.as_path())).collect()
    }

    /// True if every file exists (i.e. `prepare_datasets` has run).
    pub fn exist(&self) -> bool {
        self.tables.iter().all(|(_, p)| p.exists())
    }
}

type Builder = Box<dyn Fn(&Tables) -> Result<DataFrame> + Send + Sync>;

/// One query: a Plume plan builder and the DuckDB SQL that must produce
/// the same rows.
pub struct Workload {
    pub dataset: Dataset,
    /// Stable identifier used in result files, e.g. `tpch_q6`.
    pub name: String,
    /// Target fraction of rows matched, for the selectivity sweep.
    pub target_selectivity: Option<f64>,
    pub sql: String,
    pub technique: Technique,
    pub check: Check,
    build: Builder,
}

impl Workload {
    /// Build the unoptimized Plume plan against `tables`.
    pub fn plan(&self, tables: &Tables) -> Result<DataFrame> {
        anyhow::ensure!(
            tables.dataset == self.dataset,
            "{} is a {} query, got {} tables",
            self.name,
            self.dataset.name(),
            tables.dataset.name()
        );
        (self.build)(tables)
    }

    /// Check Plume's `actual` result. `expected` is DuckDB's result for
    /// `self.sql`; `conn` has the query's tables registered as views (for
    /// checks whose reference is too large to pull out of DuckDB).
    pub fn verify(
        &self,
        actual: &[RecordBatch],
        expected: &[RecordBatch],
        conn: &Connection,
    ) -> Result<()> {
        self.check.verify(actual, expected, conn)
    }
}

/// Every Phase 1 query, TPC-H first.
pub fn all() -> Vec<Workload> {
    let mut w = tpch();
    w.extend(taxi());
    w
}

pub fn for_dataset(dataset: Dataset) -> Vec<Workload> {
    match dataset {
        Dataset::Tpch => tpch(),
        Dataset::Taxi => taxi(),
    }
}

/// The Phase 1 queries plus the Phase 2 LIMIT, top-k and join queries.
pub fn phase2_for_dataset(dataset: Dataset) -> Vec<Workload> {
    let mut w = for_dataset(dataset);
    w.extend(match dataset {
        Dataset::Tpch => tpch_phase2(),
        Dataset::Taxi => taxi_phase2(),
    });
    w
}

// ---------------------------------------------------------------------
// TPC-H
// ---------------------------------------------------------------------

/// (label, target selectivity, lower bound inclusive, upper bound exclusive)
const TPCH_SWEEP: [(&str, f64, &str, &str); 4] = [
    ("0.1pct", 0.001, "1995-06-01", "1995-06-04"),
    ("1pct", 0.01, "1995-06-01", "1995-06-26"),
    ("10pct", 0.10, "1995-06-01", "1996-02-06"),
    ("50pct", 0.50, "1993-09-09", "1997-02-20"),
];

fn tpch() -> Vec<Workload> {
    let mut out = vec![
        // TPC-H Q6, unchanged apart from the extra `n` column. DECIMAL
        // columns are DOUBLE in these files; 0.05 and 0.07 are the doubles
        // DuckDB produces casting DECIMAL(15,2) 0.05 / 0.07, so the
        // BETWEEN bounds compare the same way in both engines.
        Workload {
            dataset: Dataset::Tpch,
            name: "tpch_q6".into(),
            target_selectivity: None,
            sql: "SELECT sum(l_extendedprice * l_discount) AS revenue, count(*) AS n \
                  FROM lineitem \
                  WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' \
                    AND l_discount >= 0.05 AND l_discount <= 0.07 \
                    AND l_quantity < 24"
                .into(),
            technique: Technique::Filter,
            check: Check::Exact,
            build: Box::new(|t| {
                Ok(t.scan("lineitem")?
                    .filter(
                        col("l_shipdate")
                            .gt_eq(lit_date("1994-01-01"))
                            .and(col("l_shipdate").lt(lit_date("1995-01-01")))
                            .and(col("l_discount").gt_eq(lit(0.05)))
                            .and(col("l_discount").lt_eq(lit(0.07)))
                            .and(col("l_quantity").lt(lit(24.0))),
                    )?
                    .aggregate(
                        vec![],
                        vec![
                            sum(col("l_extendedprice").mul(col("l_discount"))).alias("revenue"),
                            count_star().alias("n"),
                        ],
                    )?)
            }),
        },
        // TPC-H Q1 without ORDER BY. 1998-09-02 = DATE '1998-12-01' - 90 days.
        Workload {
            dataset: Dataset::Tpch,
            name: "tpch_q1".into(),
            target_selectivity: None,
            sql: "SELECT l_returnflag, l_linestatus, \
                    sum(l_quantity) AS sum_qty, \
                    sum(l_extendedprice) AS sum_base_price, \
                    sum(l_extendedprice * (1.0 - l_discount)) AS sum_disc_price, \
                    sum(l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax)) AS sum_charge, \
                    avg(l_quantity) AS avg_qty, \
                    avg(l_extendedprice) AS avg_price, \
                    avg(l_discount) AS avg_disc, \
                    count(*) AS n \
                  FROM lineitem \
                  WHERE l_shipdate <= DATE '1998-09-02' \
                  GROUP BY l_returnflag, l_linestatus"
                .into(),
            technique: Technique::Filter,
            check: Check::Exact,
            build: Box::new(|t| {
                let disc_price = || col("l_extendedprice").mul(lit(1.0).minus(col("l_discount")));
                Ok(t.scan("lineitem")?
                    .filter(col("l_shipdate").lt_eq(lit_date("1998-09-02")))?
                    .aggregate(
                        vec![col("l_returnflag"), col("l_linestatus")],
                        vec![
                            sum(col("l_quantity")).alias("sum_qty"),
                            sum(col("l_extendedprice")).alias("sum_base_price"),
                            sum(disc_price()).alias("sum_disc_price"),
                            sum(disc_price().mul(lit(1.0).plus(col("l_tax")))).alias("sum_charge"),
                            avg(col("l_quantity")).alias("avg_qty"),
                            avg(col("l_extendedprice")).alias("avg_price"),
                            avg(col("l_discount")).alias("avg_disc"),
                            count_star().alias("n"),
                        ],
                    )?)
            }),
        },
        // Q12-style join: lineitem probes, orders is the hash-join build
        // side (Plume builds on the right input). The date filter is on
        // l_receiptdate, which is not a sort key in any layout but tracks
        // l_shipdate (receipt = ship + 1..30 days), so it measures pruning
        // on a correlated column. See the module docs for what differs
        // from the real Q12.
        Workload {
            dataset: Dataset::Tpch,
            name: "tpch_q12_join".into(),
            target_selectivity: None,
            sql: "SELECT l_shipmode, o_orderpriority, count(*) AS n \
                  FROM lineitem JOIN orders ON l_orderkey = o_orderkey \
                  WHERE (l_shipmode = 'MAIL' OR l_shipmode = 'SHIP') \
                    AND l_commitdate < l_receiptdate \
                    AND l_shipdate < l_commitdate \
                    AND l_receiptdate >= DATE '1994-01-01' \
                    AND l_receiptdate < DATE '1995-01-01' \
                  GROUP BY l_shipmode, o_orderpriority"
                .into(),
            technique: Technique::Filter,
            check: Check::Exact,
            build: Box::new(|t| {
                Ok(t.scan("lineitem")?
                    .join(
                        t.scan("orders")?,
                        vec![(col("l_orderkey"), col("o_orderkey"))],
                        JoinType::Inner,
                    )?
                    .filter(
                        col("l_shipmode")
                            .eq(lit("MAIL"))
                            .or(col("l_shipmode").eq(lit("SHIP")))
                            .and(col("l_commitdate").lt(col("l_receiptdate")))
                            .and(col("l_shipdate").lt(col("l_commitdate")))
                            .and(col("l_receiptdate").gt_eq(lit_date("1994-01-01")))
                            .and(col("l_receiptdate").lt(lit_date("1995-01-01"))),
                    )?
                    .aggregate(
                        vec![col("l_shipmode"), col("o_orderpriority")],
                        vec![count_star().alias("n")],
                    )?)
            }),
        },
    ];

    for (label, target, lo, hi) in TPCH_SWEEP {
        out.push(Workload {
            dataset: Dataset::Tpch,
            name: format!("tpch_shipdate_{label}"),
            target_selectivity: Some(target),
            sql: format!(
                "SELECT count(*) AS n, sum(l_extendedprice) AS sum_price \
                 FROM lineitem \
                 WHERE l_shipdate >= DATE '{lo}' AND l_shipdate < DATE '{hi}'"
            ),
            technique: Technique::Filter,
            check: Check::Exact,
            build: Box::new(move |t| {
                Ok(t.scan("lineitem")?
                    .filter(
                        col("l_shipdate")
                            .gt_eq(lit_date(lo))
                            .and(col("l_shipdate").lt(lit_date(hi))),
                    )?
                    .aggregate(
                        vec![],
                        vec![
                            count_star().alias("n"),
                            sum(col("l_extendedprice")).alias("sum_price"),
                        ],
                    )?)
            }),
        });
    }
    out
}

// ---------------------------------------------------------------------
// NYC taxi
// ---------------------------------------------------------------------

/// (label, target selectivity, lower bound inclusive, upper bound exclusive)
const TAXI_SWEEP: [(&str, f64, &str, &str); 4] = [
    (
        "0.1pct",
        0.001,
        "2023-03-15 12:00:00",
        "2023-03-15 16:00:00",
    ),
    ("1pct", 0.01, "2023-03-15 00:00:00", "2023-03-17 00:00:00"),
    ("10pct", 0.10, "2023-03-10 00:00:00", "2023-03-28 00:00:00"),
    ("50pct", 0.50, "2023-02-01 00:00:00", "2023-05-02 00:00:00"),
];

/// `fare_amount > X`: about the top 0.3% of fares. Not a sort key in any
/// layout, so this shows what min/max pruning does on an unclustered
/// column.
const TAXI_FARE_THRESHOLD: f64 = 100.0;

/// `PULocationID = k`: zone 1 is Newark Airport, a rare pickup zone, so
/// this is a point lookup. The compound layout sorts on PULocationID only
/// within equal pickup timestamps, so expect it to help little.
const TAXI_PU_LOCATION: i64 = 1;

fn taxi() -> Vec<Workload> {
    let mut out = Vec::new();
    for (label, target, lo, hi) in TAXI_SWEEP {
        out.push(Workload {
            dataset: Dataset::Taxi,
            name: format!("taxi_pickup_{label}"),
            target_selectivity: Some(target),
            sql: format!(
                "SELECT count(*) AS n, sum(fare_amount) AS sum_fare \
                 FROM trips \
                 WHERE tpep_pickup_datetime >= TIMESTAMP '{lo}' \
                   AND tpep_pickup_datetime < TIMESTAMP '{hi}'"
            ),
            technique: Technique::Filter,
            check: Check::Exact,
            build: Box::new(move |t| {
                Ok(t.scan("trips")?
                    .filter(
                        col("tpep_pickup_datetime")
                            .gt_eq(lit_timestamp(lo))
                            .and(col("tpep_pickup_datetime").lt(lit_timestamp(hi))),
                    )?
                    .aggregate(
                        vec![],
                        vec![
                            count_star().alias("n"),
                            sum(col("fare_amount")).alias("sum_fare"),
                        ],
                    )?)
            }),
        });
    }

    out.push(Workload {
        dataset: Dataset::Taxi,
        name: "taxi_fare_gt_100".into(),
        target_selectivity: None,
        sql: format!(
            "SELECT count(*) AS n, avg(trip_distance) AS avg_distance \
             FROM trips WHERE fare_amount > {TAXI_FARE_THRESHOLD:?}"
        ),
        technique: Technique::Filter,
        check: Check::Exact,
        build: Box::new(|t| {
            Ok(t.scan("trips")?
                .filter(col("fare_amount").gt(lit(TAXI_FARE_THRESHOLD)))?
                .aggregate(
                    vec![],
                    vec![
                        count_star().alias("n"),
                        avg(col("trip_distance")).alias("avg_distance"),
                    ],
                )?)
        }),
    });

    out.push(Workload {
        dataset: Dataset::Taxi,
        name: "taxi_pulocation_eq_1".into(),
        target_selectivity: None,
        sql: format!(
            "SELECT count(*) AS n, sum(total_amount) AS sum_total \
             FROM trips WHERE PULocationID = {TAXI_PU_LOCATION}"
        ),
        technique: Technique::Filter,
        check: Check::Exact,
        build: Box::new(|t| {
            Ok(t.scan("trips")?
                .filter(col("PULocationID").eq(lit(TAXI_PU_LOCATION)))?
                .aggregate(
                    vec![],
                    vec![
                        count_star().alias("n"),
                        sum(col("total_amount")).alias("sum_total"),
                    ],
                )?)
        }),
    });
    out
}

// ---------------------------------------------------------------------
// Phase 2: LIMIT, top-k and join queries
// ---------------------------------------------------------------------

/// Rows the LIMIT queries ask for.
const LIMIT_K: usize = 100;

/// `SELECT <cols> FROM <table> WHERE <where> LIMIT k`, checked with
/// `Check::Limit`.
fn limit_workload(
    dataset: Dataset,
    name: String,
    target: f64,
    table: &'static str,
    cols: &'static [&'static str],
    where_sql: String,
    filter: impl Fn() -> plume_core::Expr + Send + Sync + 'static,
) -> Workload {
    let select = cols.join(", ");
    let unlimited_sql = format!("SELECT {select} FROM {table} WHERE {where_sql}");
    Workload {
        dataset,
        name,
        target_selectivity: Some(target),
        sql: format!("{unlimited_sql} LIMIT {LIMIT_K}"),
        technique: Technique::Limit,
        check: Check::Limit {
            k: LIMIT_K,
            unlimited_sql,
        },
        build: Box::new(move |t| {
            Ok(t.scan(table)?
                .filter(filter())?
                .project(cols.iter().map(|c| col(*c)).collect())?
                .limit(LIMIT_K))
        }),
    }
}

/// `SELECT <cols> FROM <table> ORDER BY <key> DESC LIMIT k`, checked with
/// `Check::TopK`.
fn top_k_workload(
    dataset: Dataset,
    name: &str,
    table: &'static str,
    cols: &'static [&'static str],
    key: &'static str,
    k: usize,
) -> Workload {
    let select = cols.join(", ");
    Workload {
        dataset,
        name: name.into(),
        target_selectivity: None,
        sql: format!("SELECT {select} FROM {table} ORDER BY {key} DESC LIMIT {k}"),
        technique: Technique::TopK,
        check: Check::TopK {
            keys: vec![key.to_string()],
            unlimited_sql: format!("SELECT {select} FROM {table}"),
        },
        build: Box::new(move |t| {
            Ok(t.scan(table)?
                .project(cols.iter().map(|c| col(*c)).collect())?
                .sort(vec![col(key).desc()])?
                .limit(k))
        }),
    }
}

const TPCH_LIMIT_COLS: &[&str] = &[
    "l_orderkey",
    "l_linenumber",
    "l_shipdate",
    "l_extendedprice",
];

fn tpch_phase2() -> Vec<Workload> {
    let mut out = Vec::new();
    for (label, target, lo, hi) in TPCH_SWEEP {
        out.push(limit_workload(
            Dataset::Tpch,
            format!("tpch_limit_{label}"),
            target,
            "lineitem",
            TPCH_LIMIT_COLS,
            format!("l_shipdate >= DATE '{lo}' AND l_shipdate < DATE '{hi}'"),
            move || {
                col("l_shipdate")
                    .gt_eq(lit_date(lo))
                    .and(col("l_shipdate").lt(lit_date(hi)))
            },
        ));
    }
    out.push(top_k_workload(
        Dataset::Tpch,
        "tpch_topk_extendedprice_10",
        "lineitem",
        &["l_orderkey", "l_linenumber", "l_extendedprice"],
        "l_extendedprice",
        10,
    ));
    out.push(top_k_workload(
        Dataset::Tpch,
        "tpch_topk_shipdate_100",
        "lineitem",
        &["l_orderkey", "l_linenumber", "l_shipdate"],
        "l_shipdate",
        100,
    ));
    // lineitem probes, a filtered orders is the build side. One week of
    // order dates: dbgen draws o_orderdate independently of o_orderkey,
    // so those orders' keys span almost the whole key range.
    out.push(tpch_join_workload(
        "tpch_join_orderdate_week",
        "o_orderdate >= DATE '1995-03-01' AND o_orderdate < DATE '1995-03-08'",
        || {
            col("o_orderdate")
                .gt_eq(lit_date("1995-03-01"))
                .and(col("o_orderdate").lt(lit_date("1995-03-08")))
        },
    ));
    // One customer's orders: a few keys scattered over the whole key
    // range. Min/max covers almost everything; the exact key set (sent
    // for builds up to 1,024 keys) hits only the row groups holding them.
    out.push(tpch_join_workload(
        "tpch_join_one_customer",
        "o_custkey = 1",
        || col("o_custkey").eq(lit(1i64)),
    ));
    // The first 1% of the key range: the case where the build keys are
    // contiguous, to show what the orderkey layout makes possible.
    out.push(tpch_join_workload(
        "tpch_join_orderkey_1pct",
        "o_orderkey < 60000",
        || col("o_orderkey").lt(lit(60_000i64)),
    ));
    out
}

fn tpch_join_workload(
    name: &str,
    orders_where: &'static str,
    filter: impl Fn() -> plume_core::Expr + Send + Sync + 'static,
) -> Workload {
    Workload {
        dataset: Dataset::Tpch,
        name: name.into(),
        target_selectivity: None,
        sql: format!(
            "SELECT count(*) AS n, sum(l_extendedprice) AS revenue \
             FROM lineitem JOIN orders ON l_orderkey = o_orderkey WHERE {orders_where}"
        ),
        technique: Technique::Join,
        check: Check::Exact,
        build: Box::new(move |t| {
            Ok(t.scan("lineitem")?
                .join(
                    t.scan("orders")?.filter(filter())?,
                    vec![(col("l_orderkey"), col("o_orderkey"))],
                    JoinType::Inner,
                )?
                .aggregate(
                    vec![],
                    vec![
                        count_star().alias("n"),
                        sum(col("l_extendedprice")).alias("revenue"),
                    ],
                )?)
        }),
    }
}

const TAXI_LIMIT_COLS: &[&str] = &[
    "tpep_pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
];

fn taxi_phase2() -> Vec<Workload> {
    let mut out = Vec::new();
    for (label, target, lo, hi) in TAXI_SWEEP {
        out.push(limit_workload(
            Dataset::Taxi,
            format!("taxi_limit_{label}"),
            target,
            "trips",
            TAXI_LIMIT_COLS,
            format!(
                "tpep_pickup_datetime >= TIMESTAMP '{lo}' AND tpep_pickup_datetime < TIMESTAMP '{hi}'"
            ),
            move || {
                col("tpep_pickup_datetime")
                    .gt_eq(lit_timestamp(lo))
                    .and(col("tpep_pickup_datetime").lt(lit_timestamp(hi)))
            },
        ));
    }
    out.push(top_k_workload(
        Dataset::Taxi,
        "taxi_topk_fare_10",
        "trips",
        &["tpep_pickup_datetime", "PULocationID", "fare_amount"],
        "fare_amount",
        10,
    ));
    out.push(top_k_workload(
        Dataset::Taxi,
        "taxi_topk_pickup_100",
        "trips",
        &["tpep_pickup_datetime", "PULocationID", "fare_amount"],
        "tpep_pickup_datetime",
        100,
    ));
    // trips probes, the zones of one borough are the build side. A
    // borough's zone IDs aren't contiguous (Manhattan's span 4..263), so
    // the build range covers most IDs; EWR is the single zone 1.
    for (name, borough) in [
        ("taxi_join_zones_manhattan", "Manhattan"),
        ("taxi_join_zones_ewr", "EWR"),
    ] {
        out.push(Workload {
            dataset: Dataset::Taxi,
            name: name.into(),
            target_selectivity: None,
            sql: format!(
                "SELECT count(*) AS n, sum(fare_amount) AS sum_fare \
                 FROM trips JOIN zones ON PULocationID = LocationID \
                 WHERE Borough = '{borough}'"
            ),
            technique: Technique::Join,
            check: Check::Exact,
            build: Box::new(move |t| {
                Ok(t.scan("trips")?
                    .join(
                        t.scan("zones")?.filter(col("Borough").eq(lit(borough)))?,
                        vec![(col("PULocationID"), col("LocationID"))],
                        JoinType::Inner,
                    )?
                    .aggregate(
                        vec![],
                        vec![
                            count_star().alias("n"),
                            sum(col("fare_amount")).alias("sum_fare"),
                        ],
                    )?)
            }),
        });
    }
    out
}

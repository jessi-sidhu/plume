//! Phase 1 query set: each query defined once, as a Plume `DataFrame`
//! builder plus the equivalent DuckDB SQL.
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
//! - Top-k queries (`ORDER BY ... LIMIT k`), which the Snowflake paper
//!   prunes separately (§5); that is Phase 2.
//! - Taxi queries on hour-of-day or day-of-week (needs EXTRACT/date
//!   functions).
//! - Date arithmetic in predicates (`DATE '1998-12-01' - INTERVAL 90 DAY`):
//!   the bounds are precomputed literals in both engines.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use plume_core::{avg, col, count_star, lit, lit_date, lit_timestamp, sum, DataFrame, JoinType};
use plume_io::scan_parquet_auto;

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

/// The three layouts `prepare_datasets` writes for each fact table.
pub const LAYOUTS: [&str; 3] = ["random", "sorted", "compound"];

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
    /// the fact table changes with the layout; `orders` is written once.
    pub fn new(data_dir: &Path, dataset: Dataset, layout: &str) -> Self {
        let tables = match dataset {
            Dataset::Tpch => vec![
                (
                    "lineitem",
                    data_dir.join(format!("tpch/lineitem_{layout}.parquet")),
                ),
                ("orders", data_dir.join("tpch/orders.parquet")),
            ],
            Dataset::Taxi => vec![(
                "trips",
                data_dir.join(format!("taxi/trips_{layout}.parquet")),
            )],
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

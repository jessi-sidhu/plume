//! Week-7 deliverable: every CSV pipeline kind (project / filter / aggregate /
//! group-by) now runs against the 1M-row sales.parquet fixture and matches
//! DuckDB's `read_parquet`. The schema is auto-inferred via `scan_parquet_auto`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use plume_bench::harness::assert_matches_duckdb;
use plume_core::{avg, col, count_star, lit, max, min, sum, DataFrame};
use plume_exec::collect;
use plume_io::scan_parquet_auto;

fn sales_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures/sales.parquet");
    if !p.exists() {
        eprintln!("SKIP: {} missing — run gen_fixtures first", p.display());
        return None;
    }
    Some(p)
}

fn df(path: &Path) -> DataFrame {
    scan_parquet_auto(path).expect("read sales schema")
}

fn diff_ordered(plan: DataFrame, p: &Path, sql: &str) -> Result<()> {
    let actual = collect(&plan.into_plan())?;
    assert_matches_duckdb(&actual, sql, &[], &[("sales", p)], true)
}

fn diff_unordered(plan: DataFrame, p: &Path, sql: &str) -> Result<()> {
    let actual = collect(&plan.into_plan())?;
    assert_matches_duckdb(&actual, sql, &[], &[("sales", p)], false)
}

// ---- straight scan / projection / filter -------------------------------

#[test]
fn projection_two_cols() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    diff_ordered(
        df(&p).project(vec![col("order_id"), col("price")])?,
        &p,
        "SELECT order_id, price FROM sales",
    )
}

#[test]
fn filter_then_project() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    diff_ordered(
        df(&p).filter(col("region").eq(lit("west")))?.project(vec![
            col("order_id"),
            col("region"),
            col("price"),
        ])?,
        &p,
        "SELECT order_id, region, price FROM sales WHERE region = 'west'",
    )
}

#[test]
fn filter_float_with_limit() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    diff_ordered(
        df(&p)
            .filter(col("price").gt(lit(900.0_f64)))?
            .project(vec![col("order_id"), col("price")])?
            .limit(100),
        &p,
        "SELECT order_id, price FROM sales WHERE price > 900.0 LIMIT 100",
    )
}

// ---- global aggregation ------------------------------------------------

#[test]
fn global_aggs_match_duckdb() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    diff_ordered(
        df(&p).aggregate(
            vec![],
            vec![
                count_star().alias("n"),
                sum(col("quantity")).alias("total_qty"),
                avg(col("price")).alias("avg_price"),
                min(col("price")).alias("min_price"),
                max(col("price")).alias("max_price"),
            ],
        )?,
        &p,
        "SELECT COUNT(*) AS n, SUM(quantity) AS total_qty, AVG(price) AS avg_price, \
                MIN(price) AS min_price, MAX(price) AS max_price FROM sales",
    )
}

// ---- group-by ----------------------------------------------------------

#[test]
fn group_by_region_count_and_sum() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    diff_unordered(
        df(&p).aggregate(
            vec![col("region")],
            vec![
                count_star().alias("n"),
                sum(col("quantity")).alias("total_qty"),
            ],
        )?,
        &p,
        "SELECT region, COUNT(*) AS n, SUM(quantity) AS total_qty FROM sales GROUP BY region",
    )
}

#[test]
fn group_by_category_with_filter() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    diff_unordered(
        df(&p).filter(col("price").gt(lit(500.0_f64)))?.aggregate(
            vec![col("category")],
            vec![
                count_star().alias("n"),
                avg(col("price")).alias("avg_price"),
            ],
        )?,
        &p,
        "SELECT category, COUNT(*) AS n, AVG(price) AS avg_price \
         FROM sales WHERE price > 500.0 GROUP BY category",
    )
}

#[test]
fn group_by_two_cols_min_max_price() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    diff_unordered(
        df(&p).aggregate(
            vec![col("region"), col("category")],
            vec![min(col("price")).alias("mn"), max(col("price")).alias("mx")],
        )?,
        &p,
        "SELECT region, category, MIN(price) AS mn, MAX(price) AS mx \
         FROM sales GROUP BY region, category",
    )
}

//! Week-8 deliverable: a column-pruned plan must produce identical results
//! to the unpruned plan, and both must match DuckDB. Tests run on the
//! 1M-row sales.parquet fixture.

use std::path::{Path, PathBuf};

use anyhow::Result;
use plume_bench::harness::{assert_matches_duckdb, assert_rendered_matches};
use plume_core::{avg, col, count_star, lit, sum, DataFrame, LogicalPlan};
use plume_exec::collect;
use plume_io::scan_parquet_auto;
use plume_optimizer::prune_columns;

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

/// Run a plan three ways and assert they all agree:
///  1. Unpruned through plume.collect()
///  2. Pruned through plume.collect()
///  3. DuckDB via the harness
fn diff_three_ways_unordered(plan: LogicalPlan, p: &Path, sql: &str) -> Result<()> {
    assert_rendered_matches(&plan, sql, &[], &[("sales", p)], false)?;
    let unpruned = collect(&plan)?;
    let pruned_plan = prune_columns(plan)?;
    let pruned = collect(&pruned_plan)?;
    assert_matches_duckdb(&unpruned, sql, &[], &[("sales", p)], false)?;
    assert_matches_duckdb(&pruned, sql, &[], &[("sales", p)], false)?;
    assert_rendered_matches(&pruned_plan, sql, &[], &[("sales", p)], false)?;
    Ok(())
}

fn diff_three_ways_ordered(plan: LogicalPlan, p: &Path, sql: &str) -> Result<()> {
    assert_rendered_matches(&plan, sql, &[], &[("sales", p)], true)?;
    let unpruned = collect(&plan)?;
    let pruned_plan = prune_columns(plan)?;
    let pruned = collect(&pruned_plan)?;
    assert_matches_duckdb(&unpruned, sql, &[], &[("sales", p)], true)?;
    assert_matches_duckdb(&pruned, sql, &[], &[("sales", p)], true)?;
    assert_rendered_matches(&pruned_plan, sql, &[], &[("sales", p)], true)?;
    Ok(())
}

#[test]
fn projection_alone() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = df(&p)
        .project(vec![col("category"), col("price")])?
        .into_plan();
    diff_three_ways_ordered(plan, &p, "SELECT category, price FROM sales")
}

#[test]
fn filter_then_project_two_cols() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = df(&p)
        .filter(col("region").eq(lit("west")))?
        .project(vec![col("category"), col("price")])?
        .into_plan();
    diff_three_ways_ordered(
        plan,
        &p,
        "SELECT category, price FROM sales WHERE region = 'west'",
    )
}

#[test]
fn group_by_with_two_aggregates_prunes_to_three_cols() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = df(&p)
        .aggregate(
            vec![col("region")],
            vec![
                avg(col("price")).alias("avg_price"),
                sum(col("quantity")).alias("total_qty"),
            ],
        )?
        .into_plan();
    diff_three_ways_unordered(
        plan,
        &p,
        "SELECT region, AVG(price) AS avg_price, SUM(quantity) AS total_qty \
         FROM sales GROUP BY region",
    )
}

#[test]
fn count_star_prunes_to_one_column() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = df(&p)
        .aggregate(vec![], vec![count_star().alias("n")])?
        .into_plan();
    diff_three_ways_ordered(plan, &p, "SELECT COUNT(*) AS n FROM sales")
}

#[test]
fn full_aggregate_pipeline_with_filter_and_group_by() -> Result<()> {
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = df(&p)
        .filter(col("price").gt(lit(500.0_f64)))?
        .aggregate(
            vec![col("category")],
            vec![
                count_star().alias("n"),
                avg(col("price")).alias("avg_price"),
            ],
        )?
        .into_plan();
    diff_three_ways_unordered(
        plan,
        &p,
        "SELECT category, COUNT(*) AS n, AVG(price) AS avg_price \
         FROM sales WHERE price > 500.0 GROUP BY category",
    )
}

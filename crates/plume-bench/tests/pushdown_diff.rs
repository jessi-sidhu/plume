//! Week-9 deliverable. Two kinds of tests on the 1M-row sales.parquet
//! fixture (which gen_fixtures lays out as 8 disjoint price-band row
//! groups, exactly so this test can demonstrate pruning):
//!
//! 1. Diff: optimized (pushdown + pruning) vs unoptimized vs DuckDB.
//!    The Filter still does the per-row check, so the row count after
//!    pushdown must equal the unoptimized count.
//!
//! 2. Direct measurement: scan with pushdown returns strictly fewer rows
//!    than the same scan without pushdown — proving row-group skipping
//!    actually happens.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use plume_bench::harness::assert_matches_duckdb;
use plume_core::{col, count_star, lit, sum, DataFrame, LogicalPlan};
use plume_exec::{collect, ExecutionPlan, ParquetScanExec};
use plume_io::{read_parquet_schema, scan_parquet_auto};
use plume_optimizer::optimize;

fn sales_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("fixtures/sales.parquet");
    if !p.exists() {
        eprintln!("SKIP: {} missing — run gen_fixtures first", p.display());
        return None;
    }
    Some(p)
}

fn diff_three_ways_unordered(plan: LogicalPlan, p: &Path, sql: &str) -> Result<()> {
    let unoptimized = collect(&plan)?;
    let optimized_plan = optimize(plan)?;
    let optimized = collect(&optimized_plan)?;
    assert_matches_duckdb(&unoptimized, sql, &[], &[("sales", p)], false)?;
    assert_matches_duckdb(&optimized, sql, &[], &[("sales", p)], false)?;
    Ok(())
}

fn diff_three_ways_ordered(plan: LogicalPlan, p: &Path, sql: &str) -> Result<()> {
    let unoptimized = collect(&plan)?;
    let optimized_plan = optimize(plan)?;
    let optimized = collect(&optimized_plan)?;
    assert_matches_duckdb(&unoptimized, sql, &[], &[("sales", p)], true)?;
    assert_matches_duckdb(&optimized, sql, &[], &[("sales", p)], true)?;
    Ok(())
}

#[test]
fn highly_selective_float_predicate_round_trips() -> Result<()> {
    // price > 950 leaves only the last row group (875-1000) — heavy pruning.
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = scan_parquet_auto(&p)?
        .filter(col("price").gt(lit(950.0_f64)))?
        .project(vec![col("order_id"), col("price")])?
        .into_plan();
    diff_three_ways_ordered(
        plan, &p,
        "SELECT order_id, price FROM sales WHERE price > 950.0",
    )
}

#[test]
fn medium_selective_float_predicate_round_trips() -> Result<()> {
    // price > 500 leaves row groups 4-7 (4 of 8).
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = scan_parquet_auto(&p)?
        .filter(col("price").gt(lit(500.0_f64)))?
        .aggregate(vec![col("region")], vec![count_star().alias("n")])?
        .into_plan();
    diff_three_ways_unordered(
        plan, &p,
        "SELECT region, COUNT(*) AS n FROM sales WHERE price > 500.0 GROUP BY region",
    )
}

#[test]
fn string_eq_predicate_round_trips() -> Result<()> {
    // Region rows are randomly mixed across all row groups, so this
    // probably prunes nothing — but it shouldn't break anything either.
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = scan_parquet_auto(&p)?
        .filter(col("region").eq(lit("west")))?
        .project(vec![col("order_id"), col("region"), col("price")])?
        .into_plan();
    diff_three_ways_ordered(
        plan, &p,
        "SELECT order_id, region, price FROM sales WHERE region = 'west'",
    )
}

#[test]
fn conjunction_round_trips() -> Result<()> {
    // (price > 600) AND (region = 'east')
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = scan_parquet_auto(&p)?
        .filter(col("price").gt(lit(600.0_f64)).and(col("region").eq(lit("east"))))?
        .aggregate(vec![], vec![sum(col("quantity")).alias("total_qty")])?
        .into_plan();
    diff_three_ways_ordered(
        plan, &p,
        "SELECT SUM(quantity) AS total_qty FROM sales WHERE price > 600.0 AND region = 'east'",
    )
}

#[test]
fn pushdown_demonstrably_skips_row_groups() -> Result<()> {
    // Direct construction of ParquetScanExec, bypassing the Filter so we
    // observe the raw scan output before per-row filtering. With pushdown,
    // only row groups whose stats can satisfy `price > 950` survive — that
    // is exactly the last band [875, 1000), 125,000 rows. Without pushdown,
    // we get every row (1,000,000).
    let Some(p) = sales_path() else { return Ok(()) };
    let schema = read_parquet_schema(&p)?;

    let plain: Arc<dyn ExecutionPlan> = Arc::new(ParquetScanExec::new(p.clone(), schema.clone()));
    let plain_rows: usize = plain.execute()?.map(|b| b.unwrap().num_rows()).sum();
    assert_eq!(plain_rows, 1_000_000);

    let predicate = col("price").gt(lit(950.0_f64));
    let pruned: Arc<dyn ExecutionPlan> = Arc::new(
        ParquetScanExec::new(p.clone(), schema.clone())
            .with_pushdown_predicates(vec![predicate]),
    );
    let pruned_rows: usize = pruned.execute()?.map(|b| b.unwrap().num_rows()).sum();
    // Sales fixture: 8 row groups of 125K rows each, prices in disjoint
    // bands. price > 950 keeps only band [875, 1000) -> exactly 125K rows.
    assert_eq!(
        pruned_rows, 125_000,
        "pushdown should leave exactly the [875,1000) band, got {pruned_rows}"
    );
    assert!(
        pruned_rows < plain_rows,
        "pushdown must read strictly fewer rows than a plain scan ({pruned_rows} >= {plain_rows})"
    );
    Ok(())
}

#[test]
fn missing_stats_does_not_prune_aggressively() -> Result<()> {
    // If the predicate references a column whose stats are unavailable
    // for some reason, the executor must keep that row group. The sales
    // fixture has stats for every column, so we can't easily test the
    // null-stats path without a custom fixture — assert we still get the
    // right answer for a no-pruning case (predicate on a column whose
    // value range overlaps every row group).
    let Some(p) = sales_path() else { return Ok(()) };
    let plan = scan_parquet_auto(&p)?
        // quantity is uniformly random in [1, 20) across all row groups,
        // so > 5 should match in every row group — no pruning expected.
        .filter(col("quantity").gt(lit(5_i32)))?
        .aggregate(vec![], vec![count_star().alias("n")])?
        .into_plan();
    diff_three_ways_ordered(
        plan, &p,
        "SELECT COUNT(*) AS n FROM sales WHERE quantity > 5",
    )
}

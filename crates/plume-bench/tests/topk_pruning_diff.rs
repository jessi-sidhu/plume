//! Top-k pruning end to end: results with pruning on must match the same
//! TopK with pruning off and DuckDB, and the scan must skip exactly the
//! groups that can't matter.
//!
//! The sales.parquet tests use the gen_fixtures file: 8 row groups of 125K
//! rows, price bands [0,125) ... [875,1000), so for `price DESC` the last
//! group holds every top value and the other 7 can be skipped once the
//! heap fills. They skip if the fixture is missing; the temp-file tests
//! always run.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_bench::harness::{assert_valid_sorted_result, duckdb_query, run_rendered};
use plume_core::{col, count_star, lit, DataFrame, LogicalPlan};
use plume_exec::{collect_with_options, PlannerOptions, ScanMetricsSnapshot};
use plume_io::scan_parquet_auto;
use plume_optimizer::Optimizer;

fn sales_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures/sales.parquet");
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIP: {} missing, run gen_fixtures first", p.display());
        None
    }
}

const OFF: PlannerOptions = PlannerOptions {
    top_k_pruning: false,
    join_pruning: true,
    join_key_set: true,
    top_k_stats_init: true,
};

fn run(
    plan: &LogicalPlan,
    opts: PlannerOptions,
) -> Result<(Vec<RecordBatch>, ScanMetricsSnapshot)> {
    let (batches, metrics) = collect_with_options(plan, opts)?;
    Ok((batches, metrics[0]))
}

/// Run `df` with top-k pruning on and off. Both must agree with DuckDB's
/// `sql` on the sort keys; `all_sql` is the query without ORDER BY/LIMIT.
fn check(
    path: &Path,
    df: DataFrame,
    sql: &str,
    all_sql: &str,
    keys: &[&str],
) -> Result<(Vec<RecordBatch>, ScanMetricsSnapshot, ScanMetricsSnapshot)> {
    let plan = Optimizer::default().optimize(df.into_plan())?;
    let (on, m_on) = run(&plan, PlannerOptions::default())?;
    let (off, m_off) = run(&plan, OFF)?;
    let tables = [("t", path)];
    let expected = duckdb_query(sql, &[], &tables)?;
    let all = duckdb_query(all_sql, &[], &tables)?;
    assert_valid_sorted_result(&on, &expected, keys, &all)?;
    assert_valid_sorted_result(&off, &expected, keys, &all)?;
    assert_valid_sorted_result(&run_rendered(&plan)?, &expected, keys, &all)?;
    assert_eq!(m_off.row_groups_pruned_topk, 0);
    Ok((on, m_on, m_off))
}

fn sales(path: &Path) -> DataFrame {
    scan_parquet_auto(path).unwrap()
}

#[test]
fn desc_reads_only_the_top_band() -> Result<()> {
    let Some(p) = sales_path() else {
        return Ok(());
    };
    let (_, on, off) = check(
        &p,
        sales(&p).sort(vec![col("price").desc()])?.limit(10),
        "SELECT * FROM t ORDER BY price DESC LIMIT 10",
        "SELECT * FROM t",
        &["price"],
    )?;
    assert_eq!(on.row_groups_pruned_topk, 7);
    assert_eq!(on.row_groups_read, 1);
    assert!(on.bytes_read * 7 < off.bytes_read, "{on:?} vs {off:?}");
    assert_eq!(off.row_groups_read, 8);
    Ok(())
}

#[test]
fn asc_reads_only_the_bottom_band() -> Result<()> {
    let Some(p) = sales_path() else {
        return Ok(());
    };
    let (_, on, _) = check(
        &p,
        sales(&p).sort(vec![col("price").asc()])?.limit(100),
        "SELECT * FROM t ORDER BY price ASC LIMIT 100",
        "SELECT * FROM t",
        &["price"],
    )?;
    assert_eq!(on.row_groups_pruned_topk, 7);
    assert_eq!(on.row_groups_read, 1);
    Ok(())
}

#[test]
fn prunes_through_a_filter_and_projection() -> Result<()> {
    let Some(p) = sales_path() else {
        return Ok(());
    };
    let df = sales(&p)
        .filter(col("region").eq(lit("west")))?
        .project(vec![col("order_id"), col("price"), col("quantity")])?
        .sort(vec![col("price").desc(), col("order_id").asc()])?
        .limit(50);
    let (_, on, _) = check(
        &p,
        df,
        "SELECT order_id, price, quantity FROM t WHERE region = 'west' \
         ORDER BY price DESC, order_id ASC LIMIT 50",
        "SELECT order_id, price, quantity FROM t WHERE region = 'west'",
        &["price", "order_id"],
    )?;
    assert_eq!(on.row_groups_pruned_topk, 7);
    Ok(())
}

#[test]
fn k_spanning_several_groups_still_prunes_the_rest() -> Result<()> {
    // 200K rows: all of group 7 and part of group 6, then 0-5 are skipped.
    let Some(p) = sales_path() else {
        return Ok(());
    };
    let (_, on, _) = check(
        &p,
        sales(&p).sort(vec![col("price").desc()])?.limit(200_000),
        "SELECT * FROM t ORDER BY price DESC LIMIT 200000",
        "SELECT * FROM t",
        &["price"],
    )?;
    assert_eq!(on.row_groups_pruned_topk, 6);
    assert_eq!(on.row_groups_read, 2);
    Ok(())
}

#[test]
fn must_not_use_stats_init_with_an_uncovered_filter() -> Result<()> {
    // `price * 2 < 1500` can't be pushed to the scan, so every group looks
    // FullMatch to it. A boundary from those stats would be about 999.9
    // and rule out the groups holding the real answer (prices just under
    // 750). The planner must not allow stats initialization here.
    let Some(p) = sales_path() else {
        return Ok(());
    };
    let df = sales(&p)
        .filter(col("price").mul(lit(2.0)).lt(lit(1500.0)))?
        .sort(vec![col("price").desc()])?
        .limit(10);
    check(
        &p,
        df,
        "SELECT * FROM t WHERE price * 2 < 1500 ORDER BY price DESC LIMIT 10",
        "SELECT * FROM t WHERE price * 2 < 1500",
        &["price"],
    )?;
    Ok(())
}

#[test]
fn stats_init_with_a_covered_filter_keeps_the_answer() -> Result<()> {
    // `price < 700` is pushed down: groups 0-4 are FullMatch, group 5
    // ([625,750)) partial, 6-7 pruned. Stats init may only count groups
    // 0-4, so the boundary can't exclude group 5, which holds the answer.
    let Some(p) = sales_path() else {
        return Ok(());
    };
    let df = sales(&p)
        .filter(col("price").lt(lit(700.0)))?
        .sort(vec![col("price").desc()])?
        .limit(10);
    let (_, on, _) = check(
        &p,
        df,
        "SELECT * FROM t WHERE price < 700 ORDER BY price DESC LIMIT 10",
        "SELECT * FROM t WHERE price < 700",
        &["price"],
    )?;
    assert_eq!(on.row_groups_pruned, 2);
    assert_eq!(on.row_groups_pruned_topk, 5);
    Ok(())
}

// ---- temp files: ties, NULLs, and shapes that must not prune ------------

/// Write (x, y) Int64 columns, `per_group` rows per row group.
fn write_xy(dir: &Path, x: Vec<Option<i64>>, y: Vec<i64>, per_group: usize) -> PathBuf {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("y", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(x)), Arc::new(Int64Array::from(y))],
    )
    .unwrap();
    let path = dir.join("xy.parquet");
    let props = WriterProperties::builder()
        .set_max_row_group_size(per_group)
        .build();
    let mut w = ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

fn column_i64(batches: &[RecordBatch], idx: usize) -> Vec<Option<i64>> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b
                .column(idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            (0..a.len())
                .map(|i| (!a.is_null(i)).then(|| a.value(i)))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn ties_with_the_boundary_are_never_skipped() -> Result<()> {
    // Group 0: x = 100, y = 10..20. Group 1: x = 100, y = 0..10. Group 2:
    // x = 0..10. For ORDER BY x DESC, y ASC LIMIT 5, group 0 is read first
    // (file order breaks the tie) and sets the boundary to x = 100. Group
    // 1's max is also 100: a tie, not strictly worse, so it must be read,
    // and it holds the right answer (y = 0..5). Group 2 can be skipped.
    let dir = tempfile::tempdir()?;
    let x: Vec<Option<i64>> = std::iter::repeat(Some(100))
        .take(20)
        .chain((0..10).map(Some))
        .collect();
    let y: Vec<i64> = (10..20).chain(0..10).chain(0..10).collect();
    let p = write_xy(dir.path(), x, y, 10);
    let df = scan_parquet_auto(&p)?
        .sort(vec![col("x").desc(), col("y").asc()])?
        .limit(5);
    let (on, m, _) = check(
        &p,
        df,
        "SELECT * FROM t ORDER BY x DESC, y ASC LIMIT 5",
        "SELECT * FROM t",
        &["x", "y"],
    )?;
    assert_eq!(column_i64(&on, 1), (0..5).map(Some).collect::<Vec<_>>());
    assert_eq!(m.row_groups_pruned_topk, 1);
    assert_eq!(m.row_groups_read, 2);
    Ok(())
}

#[test]
fn must_not_skip_a_group_with_nulls_under_nulls_first() {
    // Group 0: x = 100..110. Group 1: x = 0..10 with one NULL. For
    // ORDER BY x DESC NULLS FIRST LIMIT 3 the NULL is the best row. Judged
    // on its max (9) alone, group 1 would look skippable once group 0
    // filled the heap.
    let dir = tempfile::tempdir().unwrap();
    let mut x: Vec<Option<i64>> = (100..110).chain(0..10).map(Some).collect();
    x[15] = None;
    let p = write_xy(dir.path(), x, (0..20).collect(), 10);
    let df = scan_parquet_auto(&p)
        .unwrap()
        .sort(vec![col("x").desc().nulls_first()])
        .unwrap()
        .limit(3);
    let (on, m, _) = check(
        &p,
        df,
        "SELECT * FROM t ORDER BY x DESC NULLS FIRST LIMIT 3",
        "SELECT * FROM t",
        &["x"],
    )
    .unwrap();
    assert_eq!(column_i64(&on, 0), vec![None, Some(109), Some(108)]);
    assert_eq!(m.row_groups_pruned_topk, 0);
}

#[test]
fn nulls_last_groups_of_only_lower_values_are_skipped() {
    // Same file, NULLS LAST: the NULL is the worst row, so group 1 (max 9)
    // can't beat the boundary 108 and is skipped.
    let dir = tempfile::tempdir().unwrap();
    let mut x: Vec<Option<i64>> = (100..110).chain(0..10).map(Some).collect();
    x[15] = None;
    let p = write_xy(dir.path(), x, (0..20).collect(), 10);
    let df = scan_parquet_auto(&p)
        .unwrap()
        .sort(vec![col("x").desc()])
        .unwrap()
        .limit(3);
    let (_, m, _) = check(
        &p,
        df,
        "SELECT * FROM t ORDER BY x DESC NULLS LAST LIMIT 3",
        "SELECT * FROM t",
        &["x"],
    )
    .unwrap();
    assert_eq!(m.row_groups_pruned_topk, 1);
}

#[test]
fn must_not_prune_below_a_limit() -> Result<()> {
    // TopK over Limit: "the best 5 of the first 15 rows". Reading groups
    // best-first would change which 15 rows the Limit keeps.
    let dir = tempfile::tempdir()?;
    let x: Vec<Option<i64>> = (0..30).map(Some).collect();
    let p = write_xy(dir.path(), x, (0..30).collect(), 10);
    let df = scan_parquet_auto(&p)?
        .limit(15)
        .sort(vec![col("x").desc()])?
        .limit(5);
    let (on, m, _) = check(
        &p,
        df,
        "SELECT * FROM (SELECT * FROM t LIMIT 15) ORDER BY x DESC LIMIT 5",
        "SELECT * FROM (SELECT * FROM t LIMIT 15)",
        &["x"],
    )?;
    assert_eq!(
        column_i64(&on, 0),
        (10..15).rev().map(Some).collect::<Vec<_>>()
    );
    assert_eq!(m.row_groups_pruned_topk, 0);
    Ok(())
}

#[test]
fn must_not_prune_when_a_projection_redefines_the_column() -> Result<()> {
    // `x` above the projection is -x from the file, so the file's stats
    // for x say nothing about the sort key.
    let dir = tempfile::tempdir()?;
    let x: Vec<Option<i64>> = (0..30).map(Some).collect();
    let p = write_xy(dir.path(), x, (0..30).collect(), 10);
    let df = scan_parquet_auto(&p)?
        .project(vec![col("x").mul(lit(-1i64)).alias("x"), col("y")])?
        .sort(vec![col("x").desc()])?
        .limit(3);
    let (on, m, _) = check(
        &p,
        df,
        "SELECT -x AS x, y FROM t ORDER BY x DESC LIMIT 3",
        "SELECT -x AS x, y FROM t",
        &["x"],
    )?;
    assert_eq!(column_i64(&on, 0), vec![Some(0), Some(-1), Some(-2)]);
    assert_eq!(m.row_groups_pruned_topk, 0);
    Ok(())
}

#[test]
fn must_not_prune_below_an_aggregate() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let x: Vec<Option<i64>> = (0..30).map(|i| Some(i % 7)).collect();
    let p = write_xy(dir.path(), x, (0..30).collect(), 10);
    let df = scan_parquet_auto(&p)?
        .aggregate(vec![col("x")], vec![count_star().alias("n")])?
        .sort(vec![col("x").desc()])?
        .limit(2);
    let (_, m, _) = check(
        &p,
        df,
        "SELECT x, count(*) AS n FROM t GROUP BY x ORDER BY x DESC LIMIT 2",
        "SELECT x, count(*) AS n FROM t GROUP BY x",
        &["x", "n"],
    )?;
    assert_eq!(m.row_groups_pruned_topk, 0);
    Ok(())
}

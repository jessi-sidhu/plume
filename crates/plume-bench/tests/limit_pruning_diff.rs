//! LIMIT pruning end to end.
//!
//! LIMIT without ORDER BY may legally return different rows than DuckDB,
//! so results are checked with `assert_valid_limit_result`: exactly
//! min(k, matches) rows, each one present in DuckDB's un-limited result.
//!
//! The sales.parquet tests use the gen_fixtures file: 8 row groups of 125K
//! rows, prices in bands [0,125) ... [875,1000). For `price > 600` group 4
//! ([500,625)) is PartialMatch and groups 5-7 are FullMatch. They skip if
//! the fixture is missing; the temp-file tests always run.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_bench::harness::{assert_valid_limit_result, duckdb_query, run_rendered};
use plume_core::{col, lit, DataFrame, Expr};
use plume_exec::{collect_with_metrics, ScanMetricsSnapshot};
use plume_io::scan_parquet_auto;
use plume_optimizer::{
    ColumnPruning, ConstantFolding, DeadProjectionElim, LimitPushdown, Optimizer, PredicatePushdown,
};

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

/// The default optimizer minus LimitPruning: the "off" reference.
fn without_limit_pruning() -> Optimizer {
    Optimizer::with_rules(vec![
        Box::new(ConstantFolding),
        Box::new(DeadProjectionElim),
        Box::new(PredicatePushdown),
        Box::new(ColumnPruning),
        Box::new(LimitPushdown),
    ])
}

fn run(df: DataFrame, opt: &Optimizer) -> Result<(Vec<RecordBatch>, ScanMetricsSnapshot)> {
    let plan = opt.optimize(df.into_plan())?;
    let (batches, metrics) = collect_with_metrics(&plan)?;
    Ok((batches, metrics[0]))
}

/// Run `SELECT * FROM t WHERE <where_sql> LIMIT k` in Plume with LIMIT
/// pruning on and off, check both against DuckDB, return the metrics.
fn check(
    path: &Path,
    predicate: Expr,
    where_sql: &str,
    k: usize,
) -> Result<(ScanMetricsSnapshot, ScanMetricsSnapshot)> {
    let unlimited = duckdb_query(
        &format!("SELECT * FROM t WHERE {where_sql}"),
        &[],
        &[("t", path)],
    )?;
    let df = || -> Result<DataFrame> {
        Ok(scan_parquet_auto(path)?.filter(predicate.clone())?.limit(k))
    };
    let (on, m_on) = run(df()?, &Optimizer::default())?;
    let (off, m_off) = run(df()?, &without_limit_pruning())?;
    assert_valid_limit_result(&on, &unlimited, k)?;
    assert_valid_limit_result(&off, &unlimited, k)?;
    assert_valid_limit_result(&run_rendered(&df()?.into_plan())?, &unlimited, k)?;
    Ok((m_on, m_off))
}

#[test]
fn reads_one_full_group_when_it_covers_k() -> Result<()> {
    let Some(path) = sales_path() else {
        return Ok(());
    };
    let (on, off) = check(&path, col("price").gt(lit(600.0)), "price > 600", 100)?;
    // Off: reads in file order, starting with the partial group 4. Prices
    // are sorted within each group, so it decodes most of [500,625)
    // before reaching prices above 600.
    assert_eq!(off.row_groups_pruned, 4);
    assert_eq!(off.row_groups_pruned_limit, 0);
    assert!(off.rows_decoded > 75_000, "{}", off.rows_decoded);
    // On: group 5 is FullMatch with 125K rows, so it is the only group
    // planned, and its first batch already holds 100 rows.
    assert_eq!(on.row_groups_pruned, 4);
    assert_eq!(on.row_groups_pruned_limit, 3);
    assert_eq!(on.row_groups_read, 1);
    assert!(on.rows_decoded <= 8192, "{}", on.rows_decoded);
    Ok(())
}

#[test]
fn k_larger_than_all_matches_returns_every_match() -> Result<()> {
    let Some(path) = sales_path() else {
        return Ok(());
    };
    // price > 990 matches only part of the last group (well under 2M rows).
    let (on, _) = check(&path, col("price").gt(lit(990.0)), "price > 990", 2_000_000)?;
    assert_eq!(on.row_groups_pruned_limit, 0);
    Ok(())
}

#[test]
fn must_not_skip_groups_when_a_conjunct_is_not_pushed_down() -> Result<()> {
    // price > 600 AND price * 2 < 1500, i.e. 600 < price < 750: about 150K
    // rows from groups 4 and 5. The scan only sees price > 600, so groups
    // 5-7 look FullMatch. If the hint fired it would read just groups 5
    // and 6 (250K rows >= 200K), drop group 4, and return ~125K rows.
    let Some(path) = sales_path() else {
        return Ok(());
    };
    let pred = col("price")
        .gt(lit(600.0))
        .and(col("price").mul(lit(2.0)).lt(lit(1500.0)));
    let (on, _) = check(&path, pred, "price > 600 AND price * 2 < 1500", 200_000)?;
    assert_eq!(on.row_groups_pruned_limit, 0);
    Ok(())
}

/// Write `values` as one nullable Int64 column `x`, `per_group` rows per
/// row group.
fn write_x(dir: &Path, values: Vec<Option<i64>>, per_group: usize) -> PathBuf {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))]).unwrap();
    let path = dir.join("x.parquet");
    let props = WriterProperties::builder()
        .set_max_row_group_size(per_group)
        .build();
    let mut w = ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

#[test]
fn fewer_full_rows_than_k_still_returns_the_right_count() -> Result<()> {
    // Groups of 10: [0..10) partial for x >= 5 (5 matches), [10..20) full,
    // [20..30) full, [0..10) again partial. 20 full rows < k = 24, so the
    // scan reads both full groups, then the partial ones.
    let dir = tempfile::tempdir()?;
    let values: Vec<Option<i64>> = (0..10).chain(10..30).chain(0..10).map(Some).collect();
    let path = write_x(dir.path(), values, 10);
    let (on, _) = check(&path, col("x").gt_eq(lit(5i64)), "x >= 5", 24)?;
    assert_eq!(on.row_groups_full_match, 2);
    assert_eq!(on.row_groups_pruned_limit, 0);
    Ok(())
}

#[test]
fn nulls_keep_a_group_out_of_the_full_set() -> Result<()> {
    // Group 0 has a NULL, so it is PartialMatch for x > 0 even though its
    // range is [1, 9]. Only group 1 (10 rows) is FullMatch; k = 12 needs
    // the partial group too, and the NULL must never be returned.
    let dir = tempfile::tempdir()?;
    let mut values: Vec<Option<i64>> = (1..11).map(Some).collect();
    values[4] = None;
    values.extend((20..30).map(Some));
    let path = write_x(dir.path(), values, 10);
    let (on, _) = check(&path, col("x").gt(lit(0i64)), "x > 0", 12)?;
    assert_eq!(on.row_groups_full_match, 1);
    assert_eq!(on.row_groups_pruned_limit, 0);
    Ok(())
}

#[test]
fn limit_zero_reads_nothing() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = write_x(dir.path(), (0..30).map(Some).collect(), 10);
    let (on, _) = check(&path, col("x").gt_eq(lit(0i64)), "x >= 0", 0)?;
    assert_eq!(on.row_groups_read, 0);
    Ok(())
}

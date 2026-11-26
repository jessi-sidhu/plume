//! NOT, IS NULL and IS NOT NULL end to end. Each filter runs unoptimized
//! and optimized (pushdown + row-group pruning), both results must match
//! DuckDB, and the scan must skip exactly the row groups the null counts
//! allow. The plan rendered back to SQL must match too.
//!
//! The file has 4 row groups of 10 rows. Column `v` has no NULLs in
//! groups 0 and 3, only NULLs in group 1, and a mix in group 2. Column
//! `flag` cycles TRUE, FALSE, NULL in every group.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_bench::harness::{assert_batches_match, duckdb_query, run_rendered};
use plume_core::{col, lit, not, Expr};
use plume_exec::{collect, collect_with_options, PlannerOptions};
use plume_io::scan_parquet_auto;
use plume_optimizer::optimize;
use tempfile::TempDir;

const ROWS_PER_GROUP: usize = 10;

fn write_fixture(dir: &Path) -> std::path::PathBuf {
    let k: Vec<i64> = (0..40).collect();
    let v: Vec<Option<i64>> = (0..40i64)
        .map(|i| match i / 10 {
            1 => None,
            2 if i % 3 == 0 => None,
            _ => Some(i),
        })
        .collect();
    let flag: Vec<Option<bool>> = (0..40)
        .map(|i| match i % 3 {
            0 => Some(true),
            1 => Some(false),
            _ => None,
        })
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(k)),
            Arc::new(Int64Array::from(v)),
            Arc::new(BooleanArray::from(flag)),
        ],
    )
    .unwrap();
    let path = dir.join("t.parquet");
    let props = WriterProperties::builder()
        .set_max_row_group_size(ROWS_PER_GROUP)
        .build();
    let mut w = ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

/// Run `SELECT * FROM t WHERE <sql>` both ways, diff against DuckDB, and
/// return how many row groups the optimized scan skipped.
fn check(predicate: Expr, sql: &str) -> Result<u64> {
    let dir = TempDir::new()?;
    let path = write_fixture(dir.path());
    let plan = scan_parquet_auto(&path)?.filter(predicate)?.into_plan();
    let expected = duckdb_query(
        &format!("SELECT * FROM t WHERE {sql}"),
        &[],
        &[("t", path.as_path())],
    )?;

    assert_batches_match(&run_rendered(&plan)?, &expected, false)?;
    let unoptimized = collect(&plan)?;
    assert_batches_match(&unoptimized, &expected, false)?;

    let optimized_plan = optimize(plan)?;
    assert_batches_match(&run_rendered(&optimized_plan)?, &expected, false)?;
    let (optimized, metrics) = collect_with_options(&optimized_plan, PlannerOptions::default())?;
    assert_batches_match(&optimized, &expected, false)?;
    assert_eq!(metrics[0].row_groups_total, 4);
    Ok(metrics[0].row_groups_pruned)
}

#[test]
fn is_null_skips_groups_with_no_nulls() -> Result<()> {
    assert_eq!(check(col("v").is_null(), "v IS NULL")?, 2);
    Ok(())
}

#[test]
fn is_not_null_skips_all_null_groups() -> Result<()> {
    assert_eq!(check(col("v").is_not_null(), "v IS NOT NULL")?, 1);
    Ok(())
}

#[test]
fn null_check_combined_with_range() -> Result<()> {
    // v IS NOT NULL skips group 1; v < 10 skips groups 2 and 3.
    let pred = col("v").is_not_null().and(col("v").lt(lit(10i64)));
    assert_eq!(check(pred, "v IS NOT NULL AND v < 10")?, 3);
    Ok(())
}

#[test]
fn not_of_comparison_never_prunes() -> Result<()> {
    // Rows where v is NULL must drop out: NOT (NULL > 15) is NULL.
    assert_eq!(check(not(col("v").gt(lit(15i64))), "NOT (v > 15)")?, 0);
    Ok(())
}

#[test]
fn not_of_null_check_never_prunes() -> Result<()> {
    assert_eq!(check(not(col("v").is_null()), "NOT (v IS NULL)")?, 0);
    Ok(())
}

#[test]
fn not_over_true_false_and_null() -> Result<()> {
    check(not(col("flag")), "NOT flag")?;
    Ok(())
}

#[test]
fn null_checks_over_true_false_and_null() -> Result<()> {
    check(col("flag").is_null(), "flag IS NULL")?;
    check(col("flag").is_not_null(), "flag IS NOT NULL")?;
    Ok(())
}

#[test]
fn not_null_check_on_a_filter_result() -> Result<()> {
    // (flag AND v > 20) is NULL wherever either side is unknown and the
    // other doesn't force FALSE.
    check(
        col("flag").and(col("v").gt(lit(20i64))).is_null(),
        "(flag AND v > 20) IS NULL",
    )?;
    Ok(())
}

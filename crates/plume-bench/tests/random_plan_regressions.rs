//! Bugs the randomized tests found, pinned as focused DuckDB diffs.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Date32Array, Int64Array, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::ArrowWriter;
use plume_bench::harness::{assert_matches_duckdb, run_rendered};
use plume_core::{col, count_star, lit, max, min, JoinType};
use plume_exec::collect;
use plume_io::scan_parquet_auto;
use plume_optimizer::Optimizer;

/// t(k, d, t): a few rows with repeated dates/timestamps and NULLs.
fn write_t(dir: &Path) -> PathBuf {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("d", DataType::Date32, true),
        Field::new("t", DataType::Timestamp(TimeUnit::Microsecond, None), true),
    ]));
    let day = 86_400_000_000i64;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(2),
                Some(2),
                None,
                Some(3),
                Some(1),
            ])),
            Arc::new(Date32Array::from(vec![
                Some(19_000),
                Some(19_001),
                Some(19_000),
                None,
                Some(19_005),
                Some(19_001),
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(19_000 * day),
                Some(19_000 * day + 5),
                None,
                Some(19_002 * day),
                Some(19_000 * day),
                Some(18_999 * day),
            ])),
        ],
    )
    .unwrap();
    let path = dir.join("t.parquet");
    let mut w = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

fn check(df: plume_core::DataFrame, sql: &str, path: &Path) -> Result<()> {
    let raw = df.into_plan();
    for plan in [raw.clone(), Optimizer::default().optimize(raw)?] {
        let out = collect(&plan)?;
        assert_matches_duckdb(&out, sql, &[], &[("t", path)], false)?;
        assert_matches_duckdb(&run_rendered(&plan)?, sql, &[], &[("t", path)], false)?;
    }
    Ok(())
}

#[test]
fn group_by_over_empty_input_returns_no_rows() -> Result<()> {
    // Used to fail with "concat requires input of at least one array".
    let dir = tempfile::tempdir()?;
    let p = write_t(dir.path());
    check(
        scan_parquet_auto(&p)?
            .filter(col("k").gt(lit(100i64)))?
            .aggregate(vec![col("k")], vec![count_star().alias("n")])?,
        "SELECT k, count(*) AS n FROM t WHERE k > 100 GROUP BY k",
        &p,
    )
}

#[test]
fn group_by_date_and_timestamp() -> Result<()> {
    // Used to fail with "hash key over Date32" / "... Timestamp".
    let dir = tempfile::tempdir()?;
    let p = write_t(dir.path());
    check(
        scan_parquet_auto(&p)?.aggregate(vec![col("d")], vec![count_star().alias("n")])?,
        "SELECT d, count(*) AS n FROM t GROUP BY d",
        &p,
    )?;
    check(
        scan_parquet_auto(&p)?
            .aggregate(vec![col("t"), col("k")], vec![count_star().alias("n")])?,
        "SELECT t, k, count(*) AS n FROM t GROUP BY t, k",
        &p,
    )
}

#[test]
fn min_max_over_date_and_timestamp() -> Result<()> {
    // MIN/MAX over a timestamp used to be "not implemented", and inside a
    // GROUP BY it panicked instead of returning that error.
    let dir = tempfile::tempdir()?;
    let p = write_t(dir.path());
    let aggs = || {
        vec![
            min(col("d")).alias("min_d"),
            max(col("d")).alias("max_d"),
            min(col("t")).alias("min_t"),
            max(col("t")).alias("max_t"),
        ]
    };
    let sel = "min(d) AS min_d, max(d) AS max_d, min(t) AS min_t, max(t) AS max_t";
    check(
        scan_parquet_auto(&p)?.aggregate(vec![], aggs())?,
        &format!("SELECT {sel} FROM t"),
        &p,
    )?;
    check(
        scan_parquet_auto(&p)?.aggregate(vec![col("k")], aggs())?,
        &format!("SELECT k, {sel} FROM t GROUP BY k"),
        &p,
    )?;
    // All-NULL input: one row of NULLs.
    check(
        scan_parquet_auto(&p)?
            .filter(col("k").gt(lit(100i64)))?
            .aggregate(vec![], aggs())?,
        &format!("SELECT {sel} FROM t WHERE k > 100"),
        &p,
    )
}

#[test]
fn join_on_a_date_key() -> Result<()> {
    // Join keys go through the same key code as GROUP BY.
    let dir = tempfile::tempdir()?;
    let p = write_t(dir.path());
    let right = scan_parquet_auto(&p)?.project(vec![col("d").alias("d2"), col("k").alias("k2")])?;
    let raw = scan_parquet_auto(&p)?
        .join(right, vec![(col("d"), col("d2"))], JoinType::Inner)?
        .into_plan();
    let out = collect(&raw)?;
    assert_matches_duckdb(
        &run_rendered(&raw)?,
        "SELECT a.*, b.d AS d2, b.k AS k2 FROM t a JOIN t b ON a.d = b.d",
        &[],
        &[("t", &p)],
        false,
    )?;
    assert_matches_duckdb(
        &out,
        "SELECT a.*, b.d AS d2, b.k AS k2 FROM t a JOIN t b ON a.d = b.d",
        &[],
        &[("t", &p)],
        false,
    )
}

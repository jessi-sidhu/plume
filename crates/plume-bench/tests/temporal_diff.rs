//! DATE and TIMESTAMP filters: plume vs DuckDB, plus row-group pruning.
//!
//! The test file has 4,000 rows in 4 row groups of 1,000, sorted by both
//! temporal columns:
//!   - `d`  (DATE):      1994-01-01 + i/10 days
//!   - `ts` (TIMESTAMP): 1994-01-01 00:00:00 + i hours
//!
//! Every row with i % 37 == 5 is NULL in both columns. Rows 0 and 3999 are
//! non-NULL, so the file's min/max are exactly the first and last values.
//!
//! Every filter runs unoptimized (no pruning) and optimized (pushdown +
//! pruning); both must match DuckDB on the same file.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Date32Array, Int64Array, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_bench::harness::{assert_matches_duckdb, run_rendered};
use plume_core::{col, lit_date, lit_timestamp, Expr, ScalarValue};
use plume_exec::{collect, collect_with_metrics};
use plume_io::scan_parquet_auto;
use plume_optimizer::Optimizer;

const ROWS: i64 = 4_000;
const RG_ROWS: usize = 1_000;
const HOUR_US: i64 = 3_600 * 1_000_000;

fn day0() -> i32 {
    match ScalarValue::date32_from_str("1994-01-01").unwrap() {
        ScalarValue::Date32(d) => d,
        _ => unreachable!(),
    }
}

fn ts0() -> i64 {
    match ScalarValue::timestamp_micros_from_str("1994-01-01 00:00:00").unwrap() {
        ScalarValue::TimestampMicros(t) => t,
        _ => unreachable!(),
    }
}

fn is_null_row(i: i64) -> bool {
    i % 37 == 5
}

fn day_of(i: i64) -> i32 {
    day0() + (i / 10) as i32
}

fn ts_of(i: i64) -> i64 {
    ts0() + i * HOUR_US
}

fn write_file(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d", DataType::Date32, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
    ]));
    let ids: Vec<i64> = (0..ROWS).collect();
    let ds: Vec<Option<i32>> = (0..ROWS)
        .map(|i| (!is_null_row(i)).then(|| day_of(i)))
        .collect();
    let tss: Vec<Option<i64>> = (0..ROWS)
        .map(|i| (!is_null_row(i)).then(|| ts_of(i)))
        .collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Date32Array::from(ds)),
            Arc::new(TimestampMicrosecondArray::from(tss)),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_size(RG_ROWS)
        .build();
    let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

/// Literal for the file's min/max values, built from the raw integers so
/// the test doesn't depend on hand-computed date strings.
fn date_lit(days: i32) -> Expr {
    Expr::Literal(ScalarValue::Date32(days))
}
fn ts_lit(us: i64) -> Expr {
    Expr::Literal(ScalarValue::TimestampMicros(us))
}
/// SQL text for a literal. `display_form` renders `DATE '…'` /
/// `TIMESTAMP '…'`, which DuckDB parses; if it rendered the wrong value
/// the results would differ and the test would fail.
fn sql_lit(e: &Expr) -> String {
    match e {
        Expr::Literal(v) => v.display_form(),
        _ => unreachable!(),
    }
}

/// Run `filter` both ways, diff each against DuckDB, and return the
/// number of row groups the optimized scan pruned.
fn check(path: &Path, filter: Expr, where_sql: &str) -> Result<u64> {
    let sql = format!("SELECT id, d, ts FROM t WHERE {where_sql}");
    let plan = scan_parquet_auto(path)?.filter(filter)?.into_plan();

    let plain = collect(&plan)?;
    assert_matches_duckdb(&plain, &sql, &[], &[("t", path)], false)?;
    assert_matches_duckdb(&run_rendered(&plan)?, &sql, &[], &[("t", path)], false)?;

    let optimized = Optimizer::default().optimize(plan)?;
    let (batches, metrics) = collect_with_metrics(&optimized)?;
    assert_matches_duckdb(&batches, &sql, &[], &[("t", path)], false)?;
    assert_matches_duckdb(&run_rendered(&optimized)?, &sql, &[], &[("t", path)], false)?;

    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].row_groups_total, 4);
    Ok(metrics[0].row_groups_pruned)
}

fn with_file(f: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let dir = tempfile::TempDir::new()?;
    let path = dir.path().join("temporal.parquet");
    write_file(&path);
    f(&path)
}

// ---- DATE ----------------------------------------------------------------

#[test]
fn date_equality_middle() -> Result<()> {
    with_file(|p| {
        let pruned = check(
            p,
            col("d").eq(lit_date("1994-03-15")),
            "d = DATE '1994-03-15'",
        )?;
        assert_eq!(pruned, 3);
        Ok(())
    })
}

#[test]
fn date_boundaries_at_min() -> Result<()> {
    with_file(|p| {
        let min = date_lit(day_of(0));
        let m = sql_lit(&min);
        assert_eq!(check(p, col("d").eq(min.clone()), &format!("d = {m}"))?, 3);
        assert_eq!(check(p, col("d").lt(min.clone()), &format!("d < {m}"))?, 4);
        assert_eq!(
            check(p, col("d").lt_eq(min.clone()), &format!("d <= {m}"))?,
            3
        );
        assert_eq!(check(p, col("d").gt(min.clone()), &format!("d > {m}"))?, 0);
        assert_eq!(check(p, col("d").gt_eq(min), &format!("d >= {m}"))?, 0);
        Ok(())
    })
}

#[test]
fn date_boundaries_at_max() -> Result<()> {
    with_file(|p| {
        let max = date_lit(day_of(ROWS - 1));
        let m = sql_lit(&max);
        assert_eq!(check(p, col("d").eq(max.clone()), &format!("d = {m}"))?, 3);
        assert_eq!(check(p, col("d").gt(max.clone()), &format!("d > {m}"))?, 4);
        assert_eq!(
            check(p, col("d").gt_eq(max.clone()), &format!("d >= {m}"))?,
            3
        );
        assert_eq!(check(p, col("d").lt(max.clone()), &format!("d < {m}"))?, 0);
        assert_eq!(check(p, col("d").lt_eq(max), &format!("d <= {m}"))?, 0);
        Ok(())
    })
}

#[test]
fn date_range_and_outside_file() -> Result<()> {
    with_file(|p| {
        // Spans row groups 1 and 2 only (days 100..300 from day0).
        let lo = date_lit(day0() + 150);
        let hi = date_lit(day0() + 250);
        let where_sql = format!("d >= {} AND d < {}", sql_lit(&lo), sql_lit(&hi));
        assert_eq!(
            check(p, col("d").gt_eq(lo).and(col("d").lt(hi)), &where_sql)?,
            2
        );
        assert_eq!(
            check(
                p,
                col("d").lt(lit_date("1993-06-30")),
                "d < DATE '1993-06-30'"
            )?,
            4
        );
        Ok(())
    })
}

#[test]
fn date_not_equal_keeps_everything_and_drops_nulls() -> Result<()> {
    with_file(|p| {
        let pruned = check(
            p,
            col("d").neq(lit_date("1994-01-01")),
            "d <> DATE '1994-01-01'",
        )?;
        assert_eq!(pruned, 0);
        Ok(())
    })
}

#[test]
fn date_literal_on_left() -> Result<()> {
    with_file(|p| {
        let pruned = check(
            p,
            lit_date("1994-01-05").gt(col("d")),
            "DATE '1994-01-05' > d",
        )?;
        assert_eq!(pruned, 3);
        Ok(())
    })
}

// ---- TIMESTAMP -----------------------------------------------------------

#[test]
fn timestamp_boundaries_at_min() -> Result<()> {
    with_file(|p| {
        let min = ts_lit(ts_of(0));
        let m = sql_lit(&min);
        assert_eq!(
            check(p, col("ts").eq(min.clone()), &format!("ts = {m}"))?,
            3
        );
        assert_eq!(
            check(p, col("ts").lt(min.clone()), &format!("ts < {m}"))?,
            4
        );
        assert_eq!(check(p, col("ts").lt_eq(min), &format!("ts <= {m}"))?, 3);
        Ok(())
    })
}

#[test]
fn timestamp_boundaries_at_max() -> Result<()> {
    with_file(|p| {
        let max = ts_lit(ts_of(ROWS - 1));
        let m = sql_lit(&max);
        assert_eq!(
            check(p, col("ts").eq(max.clone()), &format!("ts = {m}"))?,
            3
        );
        assert_eq!(
            check(p, col("ts").gt(max.clone()), &format!("ts > {m}"))?,
            4
        );
        assert_eq!(
            check(p, col("ts").gt_eq(max.clone()), &format!("ts >= {m}"))?,
            3
        );
        assert_eq!(check(p, col("ts").lt_eq(max), &format!("ts <= {m}"))?, 0);
        Ok(())
    })
}

#[test]
fn timestamp_range_with_string_literals() -> Result<()> {
    with_file(|p| {
        // 1994-02-15 is hour 1080 (row group 1); 1994-03-01 12:30 is
        // hour 1428.5 (row group 1).
        let pruned = check(
            p,
            col("ts")
                .gt_eq(lit_timestamp("1994-02-15 00:00:00"))
                .and(col("ts").lt(lit_timestamp("1994-03-01 12:30:00"))),
            "ts >= TIMESTAMP '1994-02-15 00:00:00' AND ts < TIMESTAMP '1994-03-01 12:30:00'",
        )?;
        assert_eq!(pruned, 3);
        Ok(())
    })
}

#[test]
fn timestamp_fractional_second_literal() -> Result<()> {
    with_file(|p| {
        check(
            p,
            col("ts").gt(lit_timestamp("1994-01-02 03:00:00.5")),
            "ts > TIMESTAMP '1994-01-02 03:00:00.5'",
        )?;
        check(
            p,
            col("ts").lt_eq(lit_timestamp("1994-01-02T03:00:00")),
            "ts <= TIMESTAMP '1994-01-02 03:00:00'",
        )?;
        Ok(())
    })
}

// ---- type checking ---------------------------------------------------------

#[test]
fn date_vs_int_literal_is_a_type_error() -> Result<()> {
    with_file(|p| {
        let df = scan_parquet_auto(p)?;
        assert!(df
            .clone()
            .filter(col("d").eq(plume_core::lit(5_i32)))
            .is_err());
        assert!(df.filter(col("ts").eq(lit_date("1994-01-01"))).is_err());
        Ok(())
    })
}

#[test]
fn malformed_literals_are_rejected() {
    assert!(ScalarValue::date32_from_str("1994-13-01").is_err());
    assert!(ScalarValue::date32_from_str("not a date").is_err());
    assert!(ScalarValue::timestamp_micros_from_str("1994-01-01 25:00:00").is_err());
}

//! ORDER BY and top-k against DuckDB.
//!
//! The input is a generated Parquet file (50K rows, small row groups so
//! the scan yields many batches) with duplicate keys, NULLs, strings and
//! dates. Results are checked with `assert_valid_sorted_result`: the
//! ORDER BY columns must match DuckDB row by row, and every row returned
//! must exist in the input. Tied rows may come back in any order.
//!
//! The SQL always spells out NULLS FIRST/LAST so the test doesn't depend
//! on DuckDB's default null order.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Date32Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_bench::harness::{assert_valid_sorted_result, duckdb_query};
use plume_core::{col, lit, DataFrame, LogicalPlan, SortExpr};
use plume_exec::collect;
use plume_io::scan_parquet_auto;
use plume_optimizer::{
    ColumnPruning, ConstantFolding, DeadProjectionElim, LimitPruning, LimitPushdown, Optimizer,
    PredicatePushdown,
};

const ROWS: usize = 50_000;

/// id: 0..ROWS, unique. a: 0..100, NULL every 37th row. bucket: 0..4, so
/// heavy ties. f: spread floats, NULL every 53rd row. s: one of five
/// strings. d: a date in a ~3 year range.
fn write_input(dir: &Path) -> PathBuf {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut id = Vec::with_capacity(ROWS);
    let mut a = Vec::with_capacity(ROWS);
    let mut bucket = Vec::with_capacity(ROWS);
    let mut f = Vec::with_capacity(ROWS);
    let mut s = Vec::with_capacity(ROWS);
    let mut d = Vec::with_capacity(ROWS);
    let words = ["ash", "birch", "cedar", "elm", "oak"];
    for i in 0..ROWS {
        let r = next();
        id.push(i as i64);
        a.push((i % 37 != 0).then_some((r % 100) as i64));
        bucket.push(((r >> 8) % 4) as i64);
        f.push((i % 53 != 0).then_some(((r >> 16) % 1_000_000) as f64 / 100.0 + 0.25));
        s.push(words[((r >> 40) % 5) as usize]);
        d.push(18_000 + ((r >> 24) % 1_100) as i32);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("a", DataType::Int64, true),
        Field::new("bucket", DataType::Int64, false),
        Field::new("f", DataType::Float64, true),
        Field::new("s", DataType::Utf8, false),
        Field::new("d", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(id)),
            Arc::new(Int64Array::from(a)),
            Arc::new(Int64Array::from(bucket)),
            Arc::new(Float64Array::from(f)),
            Arc::new(StringArray::from(s)),
            Arc::new(Date32Array::from(d)),
        ],
    )
    .unwrap();
    let path = dir.join("t.parquet");
    let props = WriterProperties::builder()
        .set_max_row_group_size(4_096)
        .build();
    let mut w = ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    path
}

/// Optimize, run, and check against DuckDB. `sql` is the full query;
/// `all_sql` is the same without its LIMIT (or `None` when there is no
/// LIMIT). Returns the optimized plan for shape assertions.
fn check(
    path: &Path,
    df: DataFrame,
    sql: &str,
    all_sql: Option<&str>,
    keys: &[&str],
) -> Result<LogicalPlan> {
    let plan = Optimizer::default().optimize(df.into_plan())?;
    let actual = collect(&plan)?;
    let tables = [("t", path)];
    let expected = duckdb_query(sql, &[], &tables)?;
    let all = match all_sql {
        Some(q) => duckdb_query(q, &[], &tables)?,
        None => expected.clone(),
    };
    assert_valid_sorted_result(&actual, &expected, keys, &all)?;
    Ok(plan)
}

fn has_top_k(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::TopK { .. } => true,
        LogicalPlan::Scan { .. } => false,
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Projection { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. } => has_top_k(input),
        LogicalPlan::Join { left, right, .. } => has_top_k(left) || has_top_k(right),
    }
}

fn scan(path: &Path) -> DataFrame {
    scan_parquet_auto(path).unwrap()
}

// ---- full sort ---------------------------------------------------------

#[test]
fn order_by_asc_nulls_last() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    check(
        &p,
        scan(&p).sort(vec![col("a").asc()])?,
        "SELECT * FROM t ORDER BY a ASC NULLS LAST",
        None,
        &["a"],
    )?;
    Ok(())
}

#[test]
fn order_by_desc_nulls_last() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    check(
        &p,
        scan(&p).sort(vec![col("f").desc()])?,
        "SELECT * FROM t ORDER BY f DESC NULLS LAST",
        None,
        &["f"],
    )?;
    Ok(())
}

#[test]
fn order_by_nulls_first() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    check(
        &p,
        scan(&p).sort(vec![col("a").asc().nulls_first(), col("id").desc()])?,
        "SELECT * FROM t ORDER BY a ASC NULLS FIRST, id DESC",
        None,
        &["a", "id"],
    )?;
    Ok(())
}

#[test]
fn order_by_multiple_columns_and_directions() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    check(
        &p,
        scan(&p).sort(vec![
            col("s").asc(),
            col("d").desc(),
            col("f").asc().nulls_first(),
        ])?,
        "SELECT * FROM t ORDER BY s ASC, d DESC, f ASC NULLS FIRST",
        None,
        &["s", "d", "f"],
    )?;
    Ok(())
}

#[test]
fn order_by_an_expression() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    let df = scan(&p)
        .project(vec![col("id"), col("a").mul(lit(-3i64)).alias("neg")])?
        .sort(vec![col("neg").asc(), col("id").asc()])?;
    check(
        &p,
        df,
        "SELECT id, a * -3 AS neg FROM t ORDER BY neg ASC NULLS LAST, id ASC",
        None,
        &["neg", "id"],
    )?;
    Ok(())
}

#[test]
fn sort_after_filter_and_on_empty_input() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    check(
        &p,
        // Not `bucket = 2`: DuckDB 1.2.2's debug build trips an internal
        // assertion (compress_integral.cpp, `min_val <= input`) on that
        // query, though its result is right. Same shape, other predicate.
        scan(&p)
            .filter(col("s").eq(lit("oak")))?
            .sort(vec![col("d").asc(), col("id").asc()])?,
        "SELECT * FROM t WHERE s = 'oak' ORDER BY d, id",
        None,
        &["d", "id"],
    )?;
    let empty = scan(&p)
        .filter(col("bucket").gt(lit(10i64)))?
        .sort(vec![col("id").asc()])?;
    let out = collect(&Optimizer::default().optimize(empty.into_plan())?)?;
    assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
    Ok(())
}

// ---- top-k -------------------------------------------------------------

#[test]
fn top_k_desc_with_nulls_in_the_column() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    let plan = check(
        &p,
        scan(&p).sort(vec![col("f").desc()])?.limit(10),
        "SELECT * FROM t ORDER BY f DESC NULLS LAST LIMIT 10",
        Some("SELECT * FROM t"),
        &["f"],
    )?;
    assert!(has_top_k(&plan), "{}", plan.explain());
    Ok(())
}

#[test]
fn top_k_nulls_first_returns_the_nulls() -> Result<()> {
    // 50_000 / 37 rounds up to 1352 NULLs in `a`; k = 1000 is all NULLs.
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    check(
        &p,
        scan(&p)
            .sort(vec![col("a").asc().nulls_first()])?
            .limit(1000),
        "SELECT * FROM t ORDER BY a ASC NULLS FIRST LIMIT 1000",
        Some("SELECT * FROM t"),
        &["a"],
    )?;
    Ok(())
}

#[test]
fn top_k_with_ties_at_the_boundary() -> Result<()> {
    // bucket has 4 values over 50K rows, so the 1000th row is deep inside
    // a run of ties. Any 1000 rows with bucket = 0 are correct.
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    let plan = check(
        &p,
        scan(&p).sort(vec![col("bucket").asc()])?.limit(1000),
        "SELECT * FROM t ORDER BY bucket LIMIT 1000",
        Some("SELECT * FROM t"),
        &["bucket"],
    )?;
    assert!(has_top_k(&plan));
    Ok(())
}

#[test]
fn top_k_multi_key_through_filter_and_projection() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    let df = scan(&p)
        .filter(col("s").eq(lit("oak")))?
        .sort(vec![col("d").desc(), col("a").asc().nulls_first()])?
        .project(vec![col("id"), col("d"), col("a")])?
        .limit(250);
    let plan = check(
        &p,
        df,
        "SELECT id, d, a FROM t WHERE s = 'oak' ORDER BY d DESC, a ASC NULLS FIRST LIMIT 250",
        Some("SELECT id, d, a FROM t WHERE s = 'oak'"),
        &["d", "a"],
    )?;
    assert!(has_top_k(&plan), "{}", plan.explain());
    Ok(())
}

#[test]
fn top_k_larger_than_the_input() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    check(
        &p,
        scan(&p).sort(vec![col("id").desc()])?.limit(ROWS + 10),
        &format!("SELECT * FROM t ORDER BY id DESC LIMIT {}", ROWS + 10),
        Some("SELECT * FROM t"),
        &["id"],
    )?;
    Ok(())
}

#[test]
fn top_k_zero_returns_nothing() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    let plan =
        Optimizer::default().optimize(scan(&p).sort(vec![col("f").asc()])?.limit(0).into_plan())?;
    let out = collect(&plan)?;
    assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
    Ok(())
}

#[test]
fn top_k_agrees_with_sort_then_limit() -> Result<()> {
    // Same query with the TopK rewrite off: a full SortExec plus Limit.
    let dir = tempfile::tempdir()?;
    let p = write_input(dir.path());
    let keys = vec![col("s").desc(), col("f").asc()];
    let df = || scan(&p).sort(keys.clone()).unwrap().limit(777);
    let no_topk = Optimizer::with_rules(vec![
        Box::new(ConstantFolding),
        Box::new(DeadProjectionElim),
        Box::new(PredicatePushdown),
        Box::new(ColumnPruning),
        Box::new(LimitPushdown),
        Box::new(LimitPruning),
    ]);
    let sort_plan = no_topk.optimize(df().into_plan())?;
    assert!(!has_top_k(&sort_plan));
    let top_plan = Optimizer::default().optimize(df().into_plan())?;
    assert!(has_top_k(&top_plan));
    let by_sort = collect(&sort_plan)?;
    let by_topk = collect(&top_plan)?;
    let all = collect(&Optimizer::default().optimize(scan(&p).into_plan())?)?;
    assert_valid_sorted_result(&by_topk, &by_sort, &["s", "f"], &all)?;
    Ok(())
}

/// Keep the SortExpr import honest: keys built by hand work the same.
#[test]
fn explicit_sort_expr_matches_builder() {
    assert_eq!(SortExpr::new(col("x"), false, false), col("x").desc());
    assert_eq!(
        SortExpr::new(col("x"), true, true),
        col("x").asc().nulls_first()
    );
}

//! Week-3 deliverable: evaluator output diffed against DuckDB on the same
//! input. Each test writes a tiny fixture batch to a temp parquet, evaluates
//! plume's expression, and asserts the result matches DuckDB's SQL.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use plume_bench::harness::{assert_matches_duckdb, run_rendered, write_batch_parquet};
use plume_core::{col, lit, not, Expr};
use plume_exec::evaluate;
use plume_io::scan_parquet_auto;
use tempfile::TempDir;

/// Helper: evaluate `expr` on `batch`, wrap result as a one-column RecordBatch
/// named "r", and assert it matches `SELECT <sql_select> AS r FROM t`.
fn diff_expr(batch: &RecordBatch, expr: &Expr, sql_select: &str) -> Result<()> {
    let dir = TempDir::new()?;
    let parquet = write_batch_parquet(dir.path(), "t", batch)?;

    let result = evaluate(expr, batch)?;
    let out_schema = Arc::new(Schema::new(vec![Field::new(
        "r",
        result.data_type().clone(),
        result.is_nullable(),
    )]));
    let out_batch = RecordBatch::try_new(out_schema, vec![result])?;

    let sql = format!("SELECT {sql_select} AS r FROM t");
    let tables = [("t", parquet.as_path() as &Path)];
    assert_matches_duckdb(std::slice::from_ref(&out_batch), &sql, &[], &tables, true)?;

    // The same expression as a projection, rendered back to SQL.
    let plan = scan_parquet_auto(&parquet)?
        .project(vec![expr.clone().alias("r")])?
        .into_plan();
    assert_matches_duckdb(&run_rendered(&plan)?, &sql, &[], &tables, true)
}

// ---- fixtures ----------------------------------------------------------

fn ints_with_nulls() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(2),
                None,
                Some(4),
                Some(5),
                Some(-3),
            ])),
            Arc::new(Int32Array::from(vec![
                Some(10),
                None,
                Some(30),
                Some(40),
                Some(5),
                Some(0),
            ])),
        ],
    )
    .unwrap()
}

fn floats() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("p", DataType::Float64, true),
        Field::new("q", DataType::Float64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                Some(2.5),
                None,
                Some(4.0),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(0.5),
                Some(2.5),
                Some(3.0),
                None,
            ])),
        ],
    )
    .unwrap()
}

fn ids_int64() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![
            1_i64, 2, 3, 100, 1_000_000,
        ]))],
    )
    .unwrap()
}

fn names() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec![
            Some("alice"),
            None,
            Some("bob"),
            Some("alice"),
        ]))],
    )
    .unwrap()
}

fn bools() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Boolean, true),
        Field::new("y", DataType::Boolean, true),
    ]));
    let xs = vec![
        Some(true),
        Some(true),
        None,
        Some(false),
        Some(false),
        None,
        None,
        Some(true),
        Some(false),
    ];
    let ys = vec![
        Some(true),
        Some(false),
        Some(true),
        Some(true),
        Some(false),
        Some(false),
        None,
        None,
        None,
    ];
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(BooleanArray::from(xs)),
            Arc::new(BooleanArray::from(ys)),
        ],
    )
    .unwrap()
}

// ---- comparisons --------------------------------------------------------

#[test]
fn eq_int_column_vs_literal() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").eq(lit(2_i32)), "a = 2")
}
#[test]
fn neq_int_column_vs_literal() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").neq(lit(2_i32)), "a <> 2")
}
#[test]
fn lt_int_column_vs_literal() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").lt(lit(4_i32)), "a < 4")
}
#[test]
fn lt_eq_int_column_vs_literal() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").lt_eq(lit(4_i32)), "a <= 4")
}
#[test]
fn gt_int_column_vs_literal() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").gt(lit(2_i32)), "a > 2")
}
#[test]
fn gt_eq_int_column_vs_literal() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").gt_eq(lit(2_i32)), "a >= 2")
}

#[test]
fn eq_float_column_vs_literal() -> Result<()> {
    diff_expr(&floats(), &col("p").eq(lit(2.5_f64)), "p = 2.5")
}
#[test]
fn lt_float_column_vs_column() -> Result<()> {
    diff_expr(&floats(), &col("p").lt(col("q")), "p < q")
}

#[test]
fn eq_string_column_vs_literal() -> Result<()> {
    diff_expr(&names(), &col("name").eq(lit("alice")), "name = 'alice'")
}
#[test]
fn neq_string_column_vs_literal() -> Result<()> {
    diff_expr(&names(), &col("name").neq(lit("alice")), "name <> 'alice'")
}

#[test]
fn col_vs_col_eq_with_nulls() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").eq(col("b")), "a = b")
}

// ---- arithmetic --------------------------------------------------------

#[test]
fn add_int_column_vs_literal() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").plus(lit(1_i32)), "a + 1")
}
#[test]
fn sub_int_column_vs_column() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").minus(col("b")), "a - b")
}
#[test]
fn mul_int_column_vs_literal() -> Result<()> {
    diff_expr(&ints_with_nulls(), &col("a").mul(lit(10_i32)), "a * 10")
}
#[test]
fn div_int_column_vs_column() -> Result<()> {
    // DuckDB integer division returns DOUBLE by default; cast to keep it integer
    // so plume's Int32 / Int32 -> Int32 result matches. Both still skip nulls
    // and avoid divide-by-zero (we constructed b to have a 0 in last row, so
    // restrict to first 5 rows by filtering).
    diff_expr(
        &ints_with_nulls(),
        &col("b").div(col("a")),
        "CAST(b // a AS INTEGER)", // duckdb integer-division operator
    )
}

#[test]
fn mul_float() -> Result<()> {
    diff_expr(&floats(), &col("p").mul(lit(2.0_f64)), "p * 2.0")
}
#[test]
fn div_float_col_col() -> Result<()> {
    diff_expr(&floats(), &col("p").div(col("q")), "p / q")
}

#[test]
fn add_int64_with_literal() -> Result<()> {
    diff_expr(&ids_int64(), &col("id").plus(lit(100_i64)), "id + 100")
}

// ---- logical / Kleene --------------------------------------------------

#[test]
fn and_kleene_against_duckdb() -> Result<()> {
    diff_expr(&bools(), &col("x").and(col("y")), "x AND y")
}
#[test]
fn or_kleene_against_duckdb() -> Result<()> {
    diff_expr(&bools(), &col("x").or(col("y")), "x OR y")
}

// ---- compound ----------------------------------------------------------

#[test]
fn nested_predicate_with_nulls() -> Result<()> {
    diff_expr(
        &ints_with_nulls(),
        &col("a").gt(lit(1_i32)).and(col("b").lt(lit(50_i32))),
        "a > 1 AND b < 50",
    )
}

#[test]
fn nested_arithmetic_then_compare() -> Result<()> {
    diff_expr(
        &ints_with_nulls(),
        &col("a").plus(col("b")).gt(lit(10_i32)),
        "a + b > 10",
    )
}

#[test]
fn alias_does_not_affect_result() -> Result<()> {
    diff_expr(
        &ints_with_nulls(),
        &col("a").gt(lit(2_i32)).alias("big_enough"),
        "a > 2",
    )
}

#[test]
fn null_in_comparison_propagates() -> Result<()> {
    // Specifically: rows where b IS NULL should yield NULL in `b > 0`.
    diff_expr(&ints_with_nulls(), &col("b").gt(lit(0_i32)), "b > 0")
}

// ---- NOT / IS NULL / IS NOT NULL ----------------------------------------
// `bools()` holds TRUE, FALSE and NULL in both columns, so each test covers
// all three truth values of the operand.

#[test]
fn not_over_all_three_truth_values() -> Result<()> {
    diff_expr(&bools(), &not(col("x")), "NOT x")
}

#[test]
fn not_of_kleene_and() -> Result<()> {
    diff_expr(&bools(), &not(col("x").and(col("y"))), "NOT (x AND y)")
}

#[test]
fn not_of_comparison_with_nulls() -> Result<()> {
    diff_expr(
        &ints_with_nulls(),
        &not(col("a").gt(lit(2_i32))),
        "NOT (a > 2)",
    )
}

#[test]
fn is_null_over_all_three_truth_values() -> Result<()> {
    diff_expr(&bools(), &col("x").is_null(), "x IS NULL")
}

#[test]
fn is_not_null_over_all_three_truth_values() -> Result<()> {
    diff_expr(&bools(), &col("x").is_not_null(), "x IS NOT NULL")
}

#[test]
fn is_null_of_comparison() -> Result<()> {
    diff_expr(
        &ints_with_nulls(),
        &col("a").gt(col("b")).is_null(),
        "(a > b) IS NULL",
    )
}

#[test]
fn not_is_null_on_strings() -> Result<()> {
    diff_expr(&names(), &not(col("name").is_null()), "NOT (name IS NULL)")
}

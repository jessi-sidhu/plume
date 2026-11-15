//! Week-4 deliverable: end-to-end CSV scan + filter + projection + limit
//! pipelines diffed against DuckDB on the generated people.csv fixture.
//!
//! Skipped (with a clear message) if the fixture isn't present yet.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_bench::harness::assert_matches_duckdb;
use plume_core::{col, lit, DataFrame};
use plume_exec::collect;

fn people_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures/people.csv");
    if !p.exists() {
        eprintln!(
            "SKIP: {} missing — run `cargo run -p plume-bench --bin gen_fixtures` first",
            p.display()
        );
        return None;
    }
    Some(p)
}

fn people_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, true),
        Field::new("region", DataType::Utf8, false),
        Field::new("score", DataType::Float64, true),
    ]))
}

/// Run `df` end-to-end through plume and assert the result matches the SQL
/// run against the same CSV via DuckDB's `read_csv_auto`. Both pipelines
/// should be order-preserving (filter and projection retain input order),
/// so we use the order-sensitive harness comparison.
fn diff_pipeline(df: DataFrame, csv: &Path, sql: &str) -> Result<()> {
    let actual = collect(&df.into_plan())?;
    assert_matches_duckdb(&actual, sql, &[("people", csv)], &[], true)
}

#[test]
fn select_all_columns() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_pipeline(
        DataFrame::scan_csv(&csv, people_schema()).project(vec![
            col("id"),
            col("name"),
            col("age"),
            col("region"),
            col("score"),
        ])?,
        &csv,
        "SELECT id, name, age, region, score FROM people",
    )
}

#[test]
fn project_subset_two_columns() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_pipeline(
        DataFrame::scan_csv(&csv, people_schema()).project(vec![col("id"), col("name")])?,
        &csv,
        "SELECT id, name FROM people",
    )
}

#[test]
fn filter_age_greater_than_30() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_pipeline(
        DataFrame::scan_csv(&csv, people_schema())
            .filter(col("age").gt(lit(30_i32)))?
            .project(vec![col("id"), col("age")])?,
        &csv,
        "SELECT id, age FROM people WHERE age > 30",
    )
}

#[test]
fn filter_string_equality_against_literal() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_pipeline(
        DataFrame::scan_csv(&csv, people_schema())
            .filter(col("region").eq(lit("west")))?
            .project(vec![col("id"), col("region")])?,
        &csv,
        "SELECT id, region FROM people WHERE region = 'west'",
    )
}

#[test]
fn filter_compound_predicate_with_and() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_pipeline(
        DataFrame::scan_csv(&csv, people_schema())
            .filter(
                col("age")
                    .gt(lit(50_i32))
                    .and(col("region").eq(lit("east"))),
            )?
            .project(vec![col("id"), col("name"), col("age"), col("region")])?,
        &csv,
        "SELECT id, name, age, region FROM people WHERE age > 50 AND region = 'east'",
    )
}

#[test]
fn filter_float_comparison_with_nulls() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    // score has ~3% nulls; `score > 90.0` must skip them per three-valued logic.
    diff_pipeline(
        DataFrame::scan_csv(&csv, people_schema())
            .filter(col("score").gt(lit(90.0_f64)))?
            .project(vec![col("id"), col("score")])?,
        &csv,
        "SELECT id, score FROM people WHERE score > 90.0",
    )
}

#[test]
fn limit_after_filter() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_pipeline(
        DataFrame::scan_csv(&csv, people_schema())
            .filter(col("region").neq(lit("east")))?
            .project(vec![col("id"), col("region")])?
            .limit(50),
        &csv,
        "SELECT id, region FROM people WHERE region <> 'east' LIMIT 50",
    )
}

#[test]
fn projection_with_arithmetic() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_pipeline(
        DataFrame::scan_csv(&csv, people_schema())
            .filter(col("age").gt_eq(lit(0_i32)))? // drop NULL ages so DuckDB's CAST agrees row-for-row
            .project(vec![
                col("id"),
                col("age").plus(lit(10_i32)).alias("age_plus_10"),
            ])?,
        &csv,
        "SELECT id, age + 10 AS age_plus_10 FROM people WHERE age >= 0",
    )
}

//! Week-5 deliverable: global aggregation diffed against DuckDB on people.csv.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_bench::harness::assert_matches_duckdb;
use plume_core::{avg, col, count, count_star, lit, max, min, sum, DataFrame};
use plume_exec::collect;

fn people_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures/people.csv");
    if !p.exists() {
        eprintln!("SKIP: {} missing — run gen_fixtures first", p.display());
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

fn diff(df: DataFrame, csv: &Path, sql: &str) -> Result<()> {
    let actual = collect(&df.into_plan())?;
    assert_matches_duckdb(&actual, sql, &[("people", csv)], &[], true)
}

#[test]
fn count_star_matches_duckdb() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff(
        DataFrame::scan_csv(&csv, people_schema())
            .aggregate(vec![], vec![count_star().alias("n")])?,
        &csv,
        "SELECT COUNT(*) AS n FROM people",
    )
}

#[test]
fn count_expr_matches_duckdb() -> Result<()> {
    // age has ~5% nulls; COUNT(age) must skip them.
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff(
        DataFrame::scan_csv(&csv, people_schema())
            .aggregate(vec![], vec![count(col("age")).alias("n_age")])?,
        &csv,
        "SELECT COUNT(age) AS n_age FROM people",
    )
}

#[test]
fn sum_int_promotes_to_int64_matching_duckdb_bigint() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff(
        DataFrame::scan_csv(&csv, people_schema())
            .aggregate(vec![], vec![sum(col("age")).alias("total")])?,
        &csv,
        "SELECT SUM(age) AS total FROM people",
    )
}

#[test]
fn avg_float_matches_duckdb() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff(
        DataFrame::scan_csv(&csv, people_schema())
            .aggregate(vec![], vec![avg(col("score")).alias("avg_score")])?,
        &csv,
        "SELECT AVG(score) AS avg_score FROM people",
    )
}

#[test]
fn min_max_int_match_duckdb() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff(
        DataFrame::scan_csv(&csv, people_schema()).aggregate(
            vec![],
            vec![
                min(col("age")).alias("min_age"),
                max(col("age")).alias("max_age"),
            ],
        )?,
        &csv,
        "SELECT MIN(age) AS min_age, MAX(age) AS max_age FROM people",
    )
}

#[test]
fn min_max_strings_match_duckdb_lex_order() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff(
        DataFrame::scan_csv(&csv, people_schema()).aggregate(
            vec![],
            vec![
                min(col("name")).alias("min_name"),
                max(col("name")).alias("max_name"),
            ],
        )?,
        &csv,
        "SELECT MIN(name) AS min_name, MAX(name) AS max_name FROM people",
    )
}

#[test]
fn aggregate_after_filter_matches_duckdb() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff(
        DataFrame::scan_csv(&csv, people_schema())
            .filter(col("region").eq(lit("west")))?
            .aggregate(
                vec![],
                vec![
                    count_star().alias("n"),
                    avg(col("score")).alias("avg_score"),
                    sum(col("age")).alias("total_age"),
                ],
            )?,
        &csv,
        "SELECT COUNT(*) AS n, AVG(score) AS avg_score, SUM(age) AS total_age \
         FROM people WHERE region = 'west'",
    )
}

#[test]
fn combined_aggregates_one_pass_match_duckdb() -> Result<()> {
    // Six aggregates over the same input. Tests that we drive the input
    // stream once and update every accumulator per batch.
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff(
        DataFrame::scan_csv(&csv, people_schema()).aggregate(
            vec![],
            vec![
                count_star().alias("n"),
                count(col("age")).alias("n_age"),
                sum(col("age")).alias("total_age"),
                avg(col("score")).alias("avg_score"),
                min(col("score")).alias("min_score"),
                max(col("score")).alias("max_score"),
            ],
        )?,
        &csv,
        "SELECT COUNT(*) AS n, COUNT(age) AS n_age, SUM(age) AS total_age, \
                AVG(score) AS avg_score, MIN(score) AS min_score, MAX(score) AS max_score \
         FROM people",
    )
}

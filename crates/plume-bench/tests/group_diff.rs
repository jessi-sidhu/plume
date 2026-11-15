//! Week-6 deliverable: GROUP BY pipelines diffed against DuckDB on people.csv.
//! Hash group output order is unspecified, so all comparisons run in the
//! order-insensitive harness mode.

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

fn diff_unordered(df: DataFrame, csv: &Path, sql: &str) -> Result<()> {
    let actual = collect(&df.into_plan())?;
    assert_matches_duckdb(&actual, sql, &[("people", csv)], &[], false)
}

#[test]
fn group_by_region_count_star() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_unordered(
        DataFrame::scan_csv(&csv, people_schema())
            .aggregate(vec![col("region")], vec![count_star().alias("n")])?,
        &csv,
        "SELECT region, COUNT(*) AS n FROM people GROUP BY region",
    )
}

#[test]
fn group_by_name_with_count_and_avg_score() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_unordered(
        DataFrame::scan_csv(&csv, people_schema()).aggregate(
            vec![col("name")],
            vec![
                count_star().alias("n"),
                avg(col("score")).alias("avg_score"),
            ],
        )?,
        &csv,
        "SELECT name, COUNT(*) AS n, AVG(score) AS avg_score FROM people GROUP BY name",
    )
}

#[test]
fn group_by_two_columns_with_sum_and_avg() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_unordered(
        DataFrame::scan_csv(&csv, people_schema()).aggregate(
            vec![col("region"), col("name")],
            vec![
                sum(col("age")).alias("total_age"),
                avg(col("score")).alias("avg_score"),
            ],
        )?,
        &csv,
        "SELECT region, name, SUM(age) AS total_age, AVG(score) AS avg_score \
         FROM people GROUP BY region, name",
    )
}

#[test]
fn group_by_after_filter() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_unordered(
        DataFrame::scan_csv(&csv, people_schema())
            .filter(col("age").gt(lit(50_i32)))?
            .aggregate(
                vec![col("region")],
                vec![count_star().alias("n"), sum(col("age")).alias("total_age")],
            )?,
        &csv,
        "SELECT region, COUNT(*) AS n, SUM(age) AS total_age \
         FROM people WHERE age > 50 GROUP BY region",
    )
}

#[test]
fn group_by_min_max_strings() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_unordered(
        DataFrame::scan_csv(&csv, people_schema()).aggregate(
            vec![col("region")],
            vec![
                min(col("name")).alias("min_name"),
                max(col("name")).alias("max_name"),
            ],
        )?,
        &csv,
        "SELECT region, MIN(name) AS min_name, MAX(name) AS max_name FROM people GROUP BY region",
    )
}

#[test]
fn group_by_nullable_column_buckets_nulls_separately() -> Result<()> {
    // age has ~5% nulls; SQL puts all NULL ages in one group.
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_unordered(
        DataFrame::scan_csv(&csv, people_schema())
            .aggregate(vec![col("age")], vec![count_star().alias("n")])?,
        &csv,
        "SELECT age, COUNT(*) AS n FROM people GROUP BY age",
    )
}

#[test]
fn group_by_with_count_expr_skipping_nulls() -> Result<()> {
    let Some(csv) = people_path() else {
        return Ok(());
    };
    diff_unordered(
        DataFrame::scan_csv(&csv, people_schema()).aggregate(
            vec![col("region")],
            vec![
                count_star().alias("n_total"),
                count(col("score")).alias("n_score"), // nulls in score => smaller
                count(col("age")).alias("n_age"),
            ],
        )?,
        &csv,
        "SELECT region, COUNT(*) AS n_total, COUNT(score) AS n_score, COUNT(age) AS n_age \
         FROM people GROUP BY region",
    )
}

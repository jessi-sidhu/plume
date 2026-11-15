//! Three trivial tests that prove the DuckDB diff harness works end-to-end.
//!
//! These tests assume `cargo run -p plume-bench --bin gen_fixtures` has been
//! run at least once so that ../../fixtures/people.csv exists. If it doesn't,
//! the tests skip with a clear message rather than failing — Week-1 fixtures
//! are not a precondition for `cargo test` to pass.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use plume_bench::harness::{assert_matches_duckdb, duckdb_query};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures")
}

fn require_fixture(name: &str) -> Option<PathBuf> {
    let p = fixtures_dir().join(name);
    if !p.exists() {
        eprintln!(
            "SKIP: {} missing — run `cargo run -p plume-bench --bin gen_fixtures` first",
            p.display()
        );
        return None;
    }
    Some(p)
}

#[test]
fn harness_runs_duckdb_at_all() -> Result<()> {
    // Smoke test: bundled DuckDB starts, executes a constant query, returns Arrow.
    let batches = duckdb_query("SELECT 42 AS answer", &[], &[])?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    Ok(())
}

#[test]
fn harness_matches_handbuilt_batch_against_duckdb_constant() -> Result<()> {
    // Build the same single-row result by hand and ask the harness to diff it.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "answer",
        DataType::Int32,
        false,
    )]));
    let actual = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![42]))])?;
    assert_matches_duckdb(
        std::slice::from_ref(&actual),
        "SELECT CAST(42 AS INTEGER) AS answer",
        &[],
        &[],
        true,
    )
}

#[test]
fn harness_matches_csv_count_against_duckdb() -> Result<()> {
    let Some(csv) = require_fixture("people.csv") else {
        return Ok(());
    };

    // We don't have plume's scan operator yet, so we hand-build the expected
    // single-row result and let the harness confirm DuckDB agrees. This proves
    // the file-registration plumbing works on real data.
    let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, false)]));
    let actual = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![10_000]))])?;
    assert_matches_duckdb(
        std::slice::from_ref(&actual),
        "SELECT CAST(COUNT(*) AS INTEGER) AS n FROM people",
        &[("people", csv.as_path())],
        &[],
        true,
    )?;

    // Bonus sanity check: the region column has exactly the four values we put in.
    let regions_schema = Arc::new(Schema::new(vec![Field::new(
        "region",
        DataType::Utf8,
        false,
    )]));
    let regions_actual = RecordBatch::try_new(
        regions_schema,
        vec![Arc::new(StringArray::from(vec![
            "east", "north", "south", "west",
        ]))],
    )?;
    assert_matches_duckdb(
        std::slice::from_ref(&regions_actual),
        "SELECT DISTINCT region FROM people ORDER BY region",
        &[("people", csv.as_path())],
        &[],
        true,
    )
}

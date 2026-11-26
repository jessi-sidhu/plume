//! Every Phase 1 workload must return the same rows as DuckDB, in every
//! layout, with the optimizer on (pushdown + row-group pruning) and off.
//!
//! Needs the datasets from
//! `cargo run --release -p plume-bench --bin prepare_datasets`. If they
//! are missing (e.g. in CI) the test prints SKIP and passes. The files are
//! large, so the data tests only run in release builds (a debug build
//! prints SKIP):
//!
//!     cargo test --release -p plume-bench --test workloads_diff -- --nocapture

use std::path::PathBuf;

use anyhow::{Context, Result};
use plume_bench::harness::{assert_matches_duckdb, run_rendered};
use plume_bench::workloads::{self, Dataset, Tables, LAYOUTS};
use plume_exec::collect;
use plume_optimizer::Optimizer;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
}

fn check_dataset(dataset: Dataset) -> Result<()> {
    // Tens of millions of rows take too long in a debug build to sit
    // inside `cargo test --workspace`; run this file with --release.
    if cfg!(debug_assertions) {
        eprintln!(
            "SKIP {}: debug build, run `cargo test --release -p plume-bench --test workloads_diff`",
            dataset.name()
        );
        return Ok(());
    }
    let queries = workloads::for_dataset(dataset);
    for layout in LAYOUTS {
        let tables = Tables::new(&data_dir(), dataset, layout);
        if !tables.exist() {
            eprintln!(
                "SKIP {} {layout}: data missing, run \
                 `cargo run --release -p plume-bench --bin prepare_datasets`",
                dataset.name()
            );
            continue;
        }
        for q in &queries {
            let plan = q.plan(&tables)?.into_plan();
            let optimized = Optimizer::default().optimize(plan.clone())?;
            for (mode, p) in [("unoptimized", &plan), ("optimized", &optimized)] {
                let batches = collect(p).with_context(|| format!("{} {layout} {mode}", q.name))?;
                assert_matches_duckdb(&batches, &q.sql, &[], &tables.duckdb_views(), false)
                    .with_context(|| format!("{} {layout} {mode}", q.name))?;
            }
            assert_matches_duckdb(
                &run_rendered(&plan)?,
                &q.sql,
                &[],
                &tables.duckdb_views(),
                false,
            )
            .with_context(|| format!("{} {layout} rendered SQL", q.name))?;
            eprintln!("ok {} {layout}", q.name);
        }
    }
    Ok(())
}

#[test]
fn tpch_workloads_match_duckdb() -> Result<()> {
    check_dataset(Dataset::Tpch)
}

#[test]
fn taxi_workloads_match_duckdb() -> Result<()> {
    check_dataset(Dataset::Taxi)
}

#[test]
fn workload_names_are_unique() {
    let all = workloads::all();
    let mut names: Vec<_> = all.iter().map(|w| w.name.as_str()).collect();
    names.sort();
    let before = names.len();
    names.dedup();
    assert_eq!(before, names.len(), "duplicate workload names");
}

//! The Phase 2 queries (LIMIT, top-k, join) on every layout, checked the
//! way the benchmark runner checks them (`Workload::verify`). Phase 1's
//! queries are covered by workloads_diff.
//!
//! Needs the prepared data and a release build:
//!     cargo run --release -p plume-bench --bin prepare_datasets
//!     cargo test --release -p plume-bench --test phase2_workloads_diff

use std::path::PathBuf;

use anyhow::{Context, Result};
use plume_bench::harness::{duckdb_query, run_rendered};
use plume_bench::workloads::{self, Dataset, Tables, Technique};
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
    if cfg!(debug_assertions) {
        eprintln!("SKIP {}: debug build, run with --release", dataset.name());
        return Ok(());
    }
    let queries: Vec<_> = workloads::phase2_for_dataset(dataset)
        .into_iter()
        .filter(|w| w.technique != Technique::Filter)
        .collect();
    for layout in workloads::layouts(dataset) {
        let tables = Tables::new(&data_dir(), dataset, layout);
        if !tables.exist() {
            eprintln!(
                "SKIP {} {layout}: data missing, run prepare_datasets",
                dataset.name()
            );
            continue;
        }
        let conn = duckdb::Connection::open_in_memory()?;
        for (name, path) in tables.duckdb_views() {
            conn.execute_batch(&format!(
                "CREATE VIEW {name} AS SELECT * FROM read_parquet('{}');",
                path.display()
            ))?;
        }
        for q in &queries {
            let ctx = format!("{} {layout}", q.name);
            let plan = Optimizer::default().optimize(q.plan(&tables)?.into_plan())?;
            let actual = collect(&plan).with_context(|| ctx.clone())?;
            let expected =
                duckdb_query(&q.sql, &[], &tables.duckdb_views()).with_context(|| ctx.clone())?;
            q.verify(&actual, &expected, &conn)
                .with_context(|| ctx.clone())?;
            q.verify(&run_rendered(&plan)?, &expected, &conn)
                .with_context(|| format!("{ctx} rendered SQL"))?;
            eprintln!("ok {ctx}");
        }
    }
    Ok(())
}

#[test]
fn tpch_phase2_workloads_match_duckdb() -> Result<()> {
    check_dataset(Dataset::Tpch)
}

#[test]
fn taxi_phase2_workloads_match_duckdb() -> Result<()> {
    check_dataset(Dataset::Taxi)
}

#[test]
fn phase2_names_are_unique_and_tagged() {
    let mut names = std::collections::HashSet::new();
    for d in [Dataset::Tpch, Dataset::Taxi] {
        for w in workloads::phase2_for_dataset(d) {
            assert!(names.insert(w.name.clone()), "duplicate {}", w.name);
        }
    }
    for t in [Technique::Limit, Technique::TopK, Technique::Join] {
        for d in [Dataset::Tpch, Dataset::Taxi] {
            assert!(
                workloads::phase2_for_dataset(d)
                    .iter()
                    .any(|w| w.technique == t),
                "{} has no {} query",
                d.name(),
                t.name()
            );
        }
    }
}

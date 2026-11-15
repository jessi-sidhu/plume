//! Demo: filter + project on the 1M-row sales.parquet fixture, then print
//! the scan metrics: row groups pruned, bytes read, rows decoded.
//!
//! Run with:
//!
//!     cargo run -p plume-bench --example parquet_filter

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow::util::pretty::pretty_format_batches;
use plume_core::{col, lit};
use plume_exec::{collect_with_metrics, ExecutionPlan, ParquetScanExec, ScanMetricsSnapshot};
use plume_io::{read_parquet_schema, scan_parquet_auto};
use plume_optimizer::Optimizer;

fn main() -> Result<()> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures/sales.parquet");
    if !p.exists() {
        eprintln!(
            "fixture missing: {}\nrun: cargo run -p plume-bench --bin gen_fixtures",
            p.display()
        );
        std::process::exit(1);
    }

    let plan = scan_parquet_auto(&p)?
        .filter(col("price").gt(lit(950.0_f64)))?
        .project(vec![col("order_id"), col("price")])?
        .limit(20)
        .into_plan();

    println!("=== plan ===");
    println!("{}", plan.explain());

    let optimized = Optimizer::default().optimize(plan)?;
    println!("=== optimized plan (note pushdown=[(price > 950)] on the scan) ===");
    println!("{}", optimized.explain());

    let (batches, scan_metrics) = collect_with_metrics(&optimized)?;
    println!("=== first 20 rows of result ===");
    println!("{}", pretty_format_batches(&batches)?);
    println!("=== scan metrics for the optimized plan ===");
    for m in &scan_metrics {
        print_metrics(m);
    }
    println!("(the LIMIT stops decoding early, so rows_decoded is small)");

    // Direct evidence the pushdown actually fired: run the bare scan with
    // and without the pushdown predicate and compare its metrics.
    let schema = read_parquet_schema(&p)?;
    for (label, preds) in [
        ("without pushdown", vec![]),
        ("with pushdown", vec![col("price").gt(lit(950.0_f64))]),
    ] {
        let scan = ParquetScanExec::new(p.clone(), schema.clone()).with_pushdown_predicates(preds);
        let metrics = scan.metrics();
        let plan: Arc<dyn ExecutionPlan> = Arc::new(scan);
        for batch in plan.execute()? {
            batch?;
        }
        println!("=== bare scan {label} ===");
        print_metrics(&metrics.snapshot());
    }
    println!("(the filter still does the per-row check on surviving row groups)");
    Ok(())
}

fn print_metrics(m: &ScanMetricsSnapshot) {
    println!(
        "row groups pruned: {} of {}\n\
         bytes read:        {} of {}\n\
         rows decoded:      {}",
        m.row_groups_pruned, m.row_groups_total, m.bytes_read, m.bytes_total, m.rows_decoded
    );
}

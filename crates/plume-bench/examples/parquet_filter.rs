//! Demo: filter + project on the 1M-row sales.parquet fixture, then show how
//! many parquet row groups the optimizer's pushdown skipped.
//!
//! Run with:
//!
//!     cargo run -p plume-bench --example parquet_filter

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow::util::pretty::pretty_format_batches;
use plume_core::{col, lit};
use plume_exec::{collect, ExecutionPlan, ParquetScanExec};
use plume_io::{read_parquet_schema, scan_parquet_auto};
use plume_optimizer::Optimizer;

fn main() -> Result<()> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
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

    let batches = collect(&optimized)?;
    println!("=== first 20 rows of result ===");
    println!("{}", pretty_format_batches(&batches)?);

    // Direct evidence the pushdown actually fired: bypass the filter and
    // count raw rows read from the scan with vs without pushdown.
    let schema = read_parquet_schema(&p)?;
    let plain: Arc<dyn ExecutionPlan> = Arc::new(ParquetScanExec::new(p.clone(), schema.clone()));
    let plain_rows: usize = plain.execute()?.map(|b| b.unwrap().num_rows()).sum();

    let pushed: Arc<dyn ExecutionPlan> = Arc::new(
        ParquetScanExec::new(p, schema).with_pushdown_predicates(vec![col("price").gt(lit(950.0_f64))]),
    );
    let pushed_rows: usize = pushed.execute()?.map(|b| b.unwrap().num_rows()).sum();

    println!(
        "scan rows without pushdown: {plain_rows}\n\
         scan rows with    pushdown: {pushed_rows}\n\
         (pushdown skipped {} of 8 row groups; the filter still does the per-row check)",
        (plain_rows - pushed_rows) / 125_000
    );
    Ok(())
}

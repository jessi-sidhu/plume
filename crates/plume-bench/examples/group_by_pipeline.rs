//! Demo: full pipeline on sales.parquet — scan + filter + GROUP BY +
//! multiple aggregates. Run with:
//!
//!     cargo run -p plume-bench --example group_by_pipeline

use std::path::PathBuf;

use anyhow::Result;
use arrow::util::pretty::pretty_format_batches;
use plume_core::{avg, col, count_star, lit, max, min, sum};
use plume_exec::collect;
use plume_io::scan_parquet_auto;
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
        .filter(col("price").gt(lit(500.0_f64)))?
        .aggregate(
            vec![col("region"), col("category")],
            vec![
                count_star().alias("n"),
                sum(col("quantity")).alias("total_qty"),
                avg(col("price")).alias("avg_price"),
                min(col("price")).alias("min_price"),
                max(col("price")).alias("max_price"),
            ],
        )?
        .into_plan();

    println!("=== plan ===");
    println!("{}", plan.explain());

    let optimized = Optimizer::default().optimize(plan)?;
    println!("=== optimized plan ===");
    println!("{}", optimized.explain());
    println!(
        "  ^ note: column pruning kept only price/quantity/region/category;\n\
           predicate pushdown attached (price > 500) so the parquet reader\n\
           skips row groups whose max price is <= 500."
    );

    let batches = collect(&optimized)?;
    println!("=== result ===");
    println!("{}", pretty_format_batches(&batches)?);
    Ok(())
}

//! Demo: read the people.csv fixture, filter, group by region, and emit
//! aggregates. Run with:
//!
//!     cargo run -p plume-bench --example csv_aggregate
//!
//! Generate the fixture first:
//!
//!     cargo run -p plume-bench --bin gen_fixtures

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow::util::pretty::pretty_format_batches;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{avg, col, count_star, lit, DataFrame};
use plume_exec::collect;
use plume_optimizer::Optimizer;

fn people_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, true),
        Field::new("region", DataType::Utf8, false),
        Field::new("score", DataType::Float64, true),
    ]))
}

fn main() -> Result<()> {
    let csv = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures/people.csv");
    if !csv.exists() {
        eprintln!(
            "fixture missing: {}\nrun: cargo run -p plume-bench --bin gen_fixtures",
            csv.display()
        );
        std::process::exit(1);
    }

    // Build a DataFrame: scan -> filter age >= 18 -> GROUP BY region.
    let plan = DataFrame::scan_csv(&csv, people_schema())
        .filter(col("age").gt_eq(lit(18_i32)))?
        .aggregate(
            vec![col("region")],
            vec![
                count_star().alias("n"),
                avg(col("score")).alias("avg_score"),
            ],
        )?
        .into_plan();

    println!("=== plan ===");
    println!("{}", plan.explain());

    let optimized = Optimizer::default().optimize(plan)?;
    println!("=== optimized plan ===");
    println!("{}", optimized.explain());

    let batches = collect(&optimized)?;
    println!("=== result ===");
    println!("{}", pretty_format_batches(&batches)?);
    Ok(())
}

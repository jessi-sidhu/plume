//! Demo: build a deliberately-wasteful plan and let the 5-rule optimizer
//! grind it down. Shows the side-by-side BEFORE/AFTER plan trees so you
//! can see each pass do its job.
//!
//! Run with:
//!
//!     cargo run -p plume-bench --example optimizer_demo
//!
//! No fixture needed — this only inspects plan structure.

use std::sync::Arc;

use anyhow::Result;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{col, count_star, lit, sum, DataFrame};
use plume_optimizer::{explain_optimized, Optimizer};

fn sales_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
    ]))
}

fn main() -> Result<()> {
    // A query that exercises every rule:
    //   - constant folding:       (100 + 50) -> 150 in the filter literal
    //   - dead-projection elim:   identity SELECT * over the scan
    //   - predicate pushdown:     price > 150 attaches to the parquet scan
    //   - column pruning:         only region + price + quantity are needed
    //   - limit pushdown:         folds into the scan's row cap
    let plan = DataFrame::scan_parquet("fixtures/sales.parquet", sales_schema())
        // Identity projection — DeadProjectionElim should drop it.
        .project(vec![
            col("order_id"),
            col("category"),
            col("region"),
            col("price"),
            col("quantity"),
        ])?
        // Filter with a foldable RHS — ConstantFolding should evaluate the
        // arithmetic at plan time so PredicatePushdown sees `price > 150`.
        .filter(col("price").gt(lit(100.0_f64).plus(lit(50.0_f64))))?
        // GROUP BY only references region, price, quantity — ColumnPruning
        // should restrict the parquet scan to those three columns.
        .aggregate(
            vec![col("region")],
            vec![
                sum(col("quantity")).alias("total_qty"),
                count_star().alias("n"),
            ],
        )?
        // A trailing Limit at the top — LimitPushdown should fold it into
        // the Scan's row cap (and keep the Limit in place as a safety net).
        .limit(50)
        .into_plan();

    let optimized = Optimizer::default().optimize(plan.clone())?;

    println!("{}", explain_optimized(&plan, &optimized));

    println!();
    println!("Four of the seven rules fired here:");
    println!("  * ConstantFolding: (100 + 50) collapsed to 150 in the predicate.");
    println!("  * DeadProjectionElim: the SELECT-* projection is gone.");
    println!("  * PredicatePushdown: 'pushdown=[(price > 150)]' attached to the scan.");
    println!("  * ColumnPruning: 'projection=[region, price, quantity]' on the scan.");
    println!();
    println!("LimitPushdown intentionally didn't fold the Limit through the");
    println!("Aggregate — aggregates change row count, so taking 50 rows from");
    println!("the *scan* would silently change the query's meaning. Limit");
    println!("only folds past row-preserving nodes (Projection, nested Limit).");

    // ---- bonus: a plan shape where LimitPushdown DOES fire ----
    let lp_plan = DataFrame::scan_parquet("fixtures/sales.parquet", sales_schema())
        .project(vec![col("order_id"), col("price")])?
        .limit(50)
        .into_plan();
    let lp_optimized = Optimizer::default().optimize(lp_plan.clone())?;
    println!();
    println!("=== bonus: LimitPushdown firing on Limit -> Projection -> Scan ===");
    println!("{}", explain_optimized(&lp_plan, &lp_optimized));
    println!("Note 'limit=50' on the scan: the parquet reader will stop after");
    println!("50 rows. The outer Limit stays in the plan so CSV (which can't");
    println!("honour the hint) is still capped end-to-end.");
    Ok(())
}

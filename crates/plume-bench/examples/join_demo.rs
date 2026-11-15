//! Demo: a customers ⋈ orders pipeline showing join-aware optimization.
//!
//! Two queries:
//!
//! 1. INNER JOIN with a left-only filter — the optimizer pushes the
//!    filter into the customers scan and prunes both sides' columns
//!    independently.
//! 2. LEFT JOIN with a right-only filter — the optimizer must NOT push
//!    the filter through the join (it would discard NULL-padded rows
//!    the LEFT join is supposed to preserve). The filter stays above.
//!
//! Run with:
//!
//!     cargo run -p plume-bench --example join_demo
//!
//! Requires the customers/orders fixtures. Generate with:
//!
//!     cargo run -p plume-bench --bin gen_fixtures

use std::path::PathBuf;

use anyhow::Result;
use arrow::util::pretty::pretty_format_batches;
use plume_core::{col, lit, JoinType};
use plume_exec::collect;
use plume_io::scan_parquet_auto;
use plume_optimizer::{explain_optimized, Optimizer};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures")
        .join(name)
}

fn main() -> Result<()> {
    let customers_path = fixture("customers.parquet");
    let orders_path = fixture("orders.parquet");
    if !customers_path.exists() || !orders_path.exists() {
        eprintln!("fixtures missing — run: cargo run -p plume-bench --bin gen_fixtures");
        std::process::exit(1);
    }

    // ============================================================
    // Query 1: INNER JOIN + left-only filter + narrow projection.
    // The optimizer should push the filter to the customers scan
    // and prune each side's columns independently.
    // ============================================================
    let plan = scan_parquet_auto(&customers_path)?
        .join(
            scan_parquet_auto(&orders_path)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Inner,
        )?
        .filter(col("region").eq(lit("west")))?
        .project(vec![col("name"), col("amount")])?
        .into_plan();

    let optimized = Optimizer::default().optimize(plan.clone())?;

    println!("=== Query 1: INNER JOIN, left-only filter, narrow projection ===");
    println!("{}", explain_optimized(&plan, &optimized));
    println!();
    println!("Three rules cooperated to rewrite the plan:");
    println!("  * PredicatePushdown: split the (region = \"west\") filter");
    println!("    above the join and routed it to the customers side, where");
    println!("    it became a Filter wrapping the customers scan.");
    println!("  * Scan-pushdown (the same rule's second pass): attached");
    println!("    (region = \"west\") to the customers ParquetScan as a row-");
    println!("    group-pruning hint.");
    println!("  * ColumnPruning: split the projection [name, amount] across");
    println!("    the join. customers needs only [id, name, region]; orders");
    println!("    needs only [customer_id, amount]. Each scan is narrowed");
    println!("    independently — the join keys (id, customer_id) and the");
    println!("    filter column (region) are kept on top of the projected");
    println!("    output columns.");

    let batches = collect(&optimized)?;
    println!();
    println!("=== result (first batch) ===");
    println!("{}", pretty_format_batches(&batches)?);

    // ============================================================
    // Query 2: LEFT JOIN + right-only filter.
    // The optimizer must NOT push the right-only filter through the
    // LEFT join — that would discard the unmatched (NULL-padded)
    // right rows the LEFT join is supposed to keep.
    // ============================================================
    let plan2 = scan_parquet_auto(&customers_path)?
        .join(
            scan_parquet_auto(&orders_path)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Left,
        )?
        .filter(col("amount").gt(lit(500.0_f64)))?
        .into_plan();

    let optimized2 = Optimizer::default().optimize(plan2.clone())?;

    println!();
    println!("=== Query 2: LEFT JOIN with a right-only filter ===");
    println!("{}", explain_optimized(&plan2, &optimized2));
    println!();
    println!("The (amount > 500) filter stays above the LEFT join — it is NOT");
    println!("pushed to the orders scan. If it were pushed, the orders side");
    println!("would shrink and the LEFT join would emit *more* unmatched");
    println!("(NULL-padded) rows than the original query, with all-different");
    println!("left values. The filter would then evaluate against those NULL");
    println!("amounts and the final result would diverge from the original.");
    println!();
    println!("ColumnPruning still runs across the join — both sides keep only");
    println!("the columns reachable from the parent (in this case all of them,");
    println!("since the projection isn't narrowed). The optimizer's safety");
    println!("table for outer joins is in the source: see");
    println!("crates/plume-optimizer/src/lib.rs::push_filter_through_join.");

    Ok(())
}

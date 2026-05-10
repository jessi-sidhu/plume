//! Join correctness diffed against DuckDB on the customers + orders fixtures.
//!
//! This file covers the join types implemented so far. As outer / semi /
//! anti joins land in subsequent commits, more cases get added here.

use std::path::PathBuf;

use anyhow::Result;
use plume_bench::harness::assert_matches_duckdb;
use plume_core::{col, lit, JoinType};
use plume_exec::collect;
use plume_io::scan_parquet_auto;

fn customers_path() -> Option<PathBuf> {
    fixture("customers.parquet")
}
fn orders_path() -> Option<PathBuf> {
    fixture("orders.parquet")
}
fn fixture(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("fixtures").join(name);
    if !p.exists() {
        eprintln!("SKIP: {} missing — run gen_fixtures first", p.display());
        return None;
    }
    Some(p)
}

// ---- Inner join --------------------------------------------------------

#[test]
fn inner_join_single_key_drops_nulls_and_orphans() -> Result<()> {
    let Some(c) = customers_path() else { return Ok(()) };
    let Some(o) = orders_path() else { return Ok(()) };
    let plan = scan_parquet_auto(&c)?
        .join(
            scan_parquet_auto(&o)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Inner,
        )?
        .into_plan();
    let actual = collect(&plan)?;
    // SQL written to project the same column order plume produces:
    //   customers.* then orders.* (with the orders.id column appearing
    // second). The harness compares row strings, so column names don't
    // matter — only types and values.
    assert_matches_duckdb(
        &actual,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers INNER JOIN orders ON customers.id = orders.customer_id",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

#[test]
fn inner_join_multi_key_two_columns() -> Result<()> {
    let Some(c) = customers_path() else { return Ok(()) };
    let Some(o) = orders_path() else { return Ok(()) };
    // (id = customer_id) AND (region = region) — the second predicate
    // tightens the match: only orders whose region matches the
    // customer's region survive.
    let plan = scan_parquet_auto(&c)?
        .join(
            scan_parquet_auto(&o)?,
            vec![
                (col("id"), col("customer_id")),
                (col("region"), col("region")),
            ],
            JoinType::Inner,
        )?
        .into_plan();
    let actual = collect(&plan)?;
    assert_matches_duckdb(
        &actual,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers INNER JOIN orders \
           ON customers.id = orders.customer_id \
          AND customers.region = orders.region",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

#[test]
fn inner_join_then_filter_above() -> Result<()> {
    // Filter on a right-side column applied above the join. The optimizer
    // can't push it through yet (Phase 5), but the operators must handle
    // a Filter -> Join chain end-to-end.
    let Some(c) = customers_path() else { return Ok(()) };
    let Some(o) = orders_path() else { return Ok(()) };
    let plan = scan_parquet_auto(&c)?
        .join(
            scan_parquet_auto(&o)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Inner,
        )?
        .filter(col("amount").gt(lit(500.0_f64)))?
        .into_plan();
    let actual = collect(&plan)?;
    assert_matches_duckdb(
        &actual,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers INNER JOIN orders ON customers.id = orders.customer_id \
         WHERE orders.amount > 500.0",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

// ---- Cross join --------------------------------------------------------

#[test]
fn cross_join_small_slices() -> Result<()> {
    let Some(c) = customers_path() else { return Ok(()) };
    let Some(o) = orders_path() else { return Ok(()) };
    // Limit both sides before crossing so the test stays fast. 5 x 5 = 25.
    let plan = scan_parquet_auto(&c)?
        .limit(5)
        .cross_join(scan_parquet_auto(&o)?.limit(5))?
        .into_plan();
    let actual = collect(&plan)?;
    // DuckDB equivalent: subqueries with the same LIMIT, then a CROSS JOIN
    // on the limited results. ORDER BY in the subqueries to make the limit
    // deterministic — files are read in the same on-disk order.
    assert_matches_duckdb(
        &actual,
        "SELECT c.id, c.name, c.region, o.id, o.customer_id, o.region, o.amount \
         FROM (SELECT * FROM customers LIMIT 5) c \
         CROSS JOIN (SELECT * FROM orders LIMIT 5) o",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

#[test]
fn cross_join_with_residual_filter_above() -> Result<()> {
    let Some(c) = customers_path() else { return Ok(()) };
    let Some(o) = orders_path() else { return Ok(()) };
    let plan = scan_parquet_auto(&c)?
        .limit(10)
        .cross_join(scan_parquet_auto(&o)?.limit(10))?
        .filter(col("amount").gt(lit(700.0_f64)))?
        .into_plan();
    let actual = collect(&plan)?;
    assert_matches_duckdb(
        &actual,
        "SELECT c.id, c.name, c.region, o.id, o.customer_id, o.region, o.amount \
         FROM (SELECT * FROM customers LIMIT 10) c \
         CROSS JOIN (SELECT * FROM orders LIMIT 10) o \
         WHERE o.amount > 700.0",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

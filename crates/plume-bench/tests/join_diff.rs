//! Join correctness diffed against DuckDB on the customers + orders fixtures.
//!
//! This file covers the join types implemented so far. As outer / semi /
//! anti joins land in subsequent commits, more cases get added here.

use std::path::PathBuf;

use anyhow::Result;
use plume_bench::harness::{assert_matches_duckdb, run_rendered};
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
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures")
        .join(name);
    if !p.exists() {
        eprintln!("SKIP: {} missing — run gen_fixtures first", p.display());
        return None;
    }
    Some(p)
}

// ---- Inner join --------------------------------------------------------

#[test]
fn inner_join_single_key_drops_nulls_and_orphans() -> Result<()> {
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
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
        &run_rendered(&plan)?,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers INNER JOIN orders ON customers.id = orders.customer_id",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
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
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
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
        &run_rendered(&plan)?,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers INNER JOIN orders \
           ON customers.id = orders.customer_id \
          AND customers.region = orders.region",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
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
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
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
        &run_rendered(&plan)?,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers INNER JOIN orders ON customers.id = orders.customer_id \
         WHERE orders.amount > 500.0",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
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

// ---- Outer joins -------------------------------------------------------

#[test]
fn left_join_keeps_unmatched_left_rows() -> Result<()> {
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
    let plan = scan_parquet_auto(&c)?
        .join(
            scan_parquet_auto(&o)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Left,
        )?
        .into_plan();
    let actual = collect(&plan)?;
    assert_matches_duckdb(
        &run_rendered(&plan)?,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers LEFT JOIN orders ON customers.id = orders.customer_id",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
    assert_matches_duckdb(
        &actual,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers LEFT JOIN orders ON customers.id = orders.customer_id",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

#[test]
fn right_join_keeps_unmatched_right_rows() -> Result<()> {
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
    let plan = scan_parquet_auto(&c)?
        .join(
            scan_parquet_auto(&o)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Right,
        )?
        .into_plan();
    let actual = collect(&plan)?;
    assert_matches_duckdb(
        &run_rendered(&plan)?,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers RIGHT JOIN orders ON customers.id = orders.customer_id",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
    assert_matches_duckdb(
        &actual,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers RIGHT JOIN orders ON customers.id = orders.customer_id",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

#[test]
fn full_join_keeps_both_unmatched_sides() -> Result<()> {
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
    let plan = scan_parquet_auto(&c)?
        .join(
            scan_parquet_auto(&o)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Full,
        )?
        .into_plan();
    let actual = collect(&plan)?;
    assert_matches_duckdb(
        &run_rendered(&plan)?,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers FULL OUTER JOIN orders ON customers.id = orders.customer_id",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
    assert_matches_duckdb(
        &actual,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers FULL OUTER JOIN orders ON customers.id = orders.customer_id",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

#[test]
fn left_join_then_filter_on_right_amount_strips_nulls() -> Result<()> {
    // Filter sits ABOVE the join. A row from the LEFT side that didn't
    // match an order produces orders.amount = NULL, and `NULL > 100`
    // is NULL (not true), so the filter drops it. The unoptimized
    // pipeline still gives the same answer DuckDB does — that's what
    // we check here. Phase 5's pushdown rule must NOT push this
    // predicate through the join (a regression test for that rule
    // lives in optimizer/tests/, not here).
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
    let plan = scan_parquet_auto(&c)?
        .join(
            scan_parquet_auto(&o)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Left,
        )?
        .filter(col("amount").gt(lit(100.0_f64)))?
        .into_plan();
    let actual = collect(&plan)?;
    assert_matches_duckdb(
        &run_rendered(&plan)?,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers LEFT JOIN orders ON customers.id = orders.customer_id \
         WHERE orders.amount > 100.0",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
    assert_matches_duckdb(
        &actual,
        "SELECT customers.id, customers.name, customers.region, \
                orders.id, orders.customer_id, orders.region, orders.amount \
         FROM customers LEFT JOIN orders ON customers.id = orders.customer_id \
         WHERE orders.amount > 100.0",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

// ---- Semi / Anti -------------------------------------------------------

#[test]
fn semi_join_emits_only_matched_left_rows() -> Result<()> {
    // SEMI: customers that have at least one order. No duplicate
    // customer rows even when a customer has many orders. Output is
    // exactly the customers schema.
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
    let plan = scan_parquet_auto(&c)?
        .join(
            scan_parquet_auto(&o)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Semi,
        )?
        .into_plan();
    let actual = collect(&plan)?;
    // DuckDB equivalent: WHERE EXISTS subquery. Returns one row per
    // qualifying customer.
    assert_matches_duckdb(
        &run_rendered(&plan)?,
        "SELECT customers.id, customers.name, customers.region \
         FROM customers \
         WHERE EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.id)",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
    assert_matches_duckdb(
        &actual,
        "SELECT customers.id, customers.name, customers.region \
         FROM customers \
         WHERE EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.id)",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

#[test]
fn anti_join_emits_unmatched_left_rows() -> Result<()> {
    // ANTI: customers with NO orders, plus customers with NULL ids
    // (NULL ≠ NULL means they never match anything).
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
    let plan = scan_parquet_auto(&c)?
        .join(
            scan_parquet_auto(&o)?,
            vec![(col("id"), col("customer_id"))],
            JoinType::Anti,
        )?
        .into_plan();
    let actual = collect(&plan)?;
    assert_matches_duckdb(
        &run_rendered(&plan)?,
        "SELECT customers.id, customers.name, customers.region \
         FROM customers \
         WHERE NOT EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.id)",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
    assert_matches_duckdb(
        &actual,
        "SELECT customers.id, customers.name, customers.region \
         FROM customers \
         WHERE NOT EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.id)",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )
}

// ---- Cross join --------------------------------------------------------

#[test]
fn cross_join_small_slices() -> Result<()> {
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
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
        &run_rendered(&plan)?,
        "SELECT c.id, c.name, c.region, o.id, o.customer_id, o.region, o.amount \
         FROM (SELECT * FROM customers LIMIT 5) c \
         CROSS JOIN (SELECT * FROM orders LIMIT 5) o",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
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
    let Some(c) = customers_path() else {
        return Ok(());
    };
    let Some(o) = orders_path() else {
        return Ok(());
    };
    let plan = scan_parquet_auto(&c)?
        .limit(10)
        .cross_join(scan_parquet_auto(&o)?.limit(10))?
        .filter(col("amount").gt(lit(700.0_f64)))?
        .into_plan();
    let actual = collect(&plan)?;
    assert_matches_duckdb(
        &run_rendered(&plan)?,
        "SELECT c.id, c.name, c.region, o.id, o.customer_id, o.region, o.amount \
         FROM (SELECT * FROM customers LIMIT 10) c \
         CROSS JOIN (SELECT * FROM orders LIMIT 10) o \
         WHERE o.amount > 700.0",
        &[],
        &[("customers", &c), ("orders", &o)],
        false,
    )?;
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

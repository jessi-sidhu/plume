//! Join pruning end to end. Each query runs with join pruning on and off;
//! both results must match DuckDB, and the probe scan's
//! `row_groups_pruned_join` must be exactly what the build keys allow.
//!
//! The probe file `facts` has `k` = 0..1000 in order, 100 rows per row
//! group, so group g holds k in [100g, 100g + 99]. The build file `dims`
//! varies per test. Plume builds on the right input, so `facts` is always
//! the left (probe) side.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_bench::harness::{assert_batches_match, duckdb_query};
use plume_core::{col, lit, DataFrame, JoinType, LogicalPlan};
use plume_exec::{collect_with_options, PlannerOptions, ScanMetricsSnapshot};
use plume_io::scan_parquet_auto;
use plume_optimizer::Optimizer;

const OFF: PlannerOptions = PlannerOptions {
    top_k_pruning: true,
    join_pruning: false,
    join_key_set: true,
};

/// Join pruning with min/max ranges only, no exact key sets.
const RANGE: PlannerOptions = PlannerOptions {
    top_k_pruning: true,
    join_pruning: true,
    join_key_set: false,
};

struct Files {
    _dir: tempfile::TempDir,
    facts: PathBuf,
    dims: PathBuf,
}

/// facts(k, v): k = 0..1000, v = "v{k}". dims(dk, label): one row per
/// entry of `dims`.
fn files(dims: Vec<Option<i64>>) -> Files {
    let dir = tempfile::tempdir().unwrap();
    let facts = dir.path().join("facts.parquet");
    let k: Vec<i64> = (0..1000).collect();
    let v: Vec<String> = k.iter().map(|k| format!("v{k}")).collect();
    write(
        &facts,
        vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Utf8, false),
        ],
        vec![
            Arc::new(Int64Array::from(k)),
            Arc::new(StringArray::from(v)),
        ],
        100,
    );
    let path = dir.path().join("dims.parquet");
    let labels: Vec<String> = (0..dims.len()).map(|i| format!("d{i}")).collect();
    write(
        &path,
        vec![
            Field::new("dk", DataType::Int64, true),
            Field::new("label", DataType::Utf8, false),
        ],
        vec![
            Arc::new(Int64Array::from(dims)),
            Arc::new(StringArray::from(labels)),
        ],
        1000,
    );
    Files {
        _dir: dir,
        facts,
        dims: path,
    }
}

fn write(path: &Path, fields: Vec<Field>, cols: Vec<arrow_array::ArrayRef>, per_group: usize) {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_size(per_group)
        .build();
    let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn run(
    plan: &LogicalPlan,
    opts: PlannerOptions,
) -> Result<(Vec<RecordBatch>, ScanMetricsSnapshot)> {
    let (batches, metrics) = collect_with_options(plan, opts)?;
    // The probe (left) scan is built first.
    Ok((batches, metrics[0]))
}

/// Run with join pruning on, on with min/max only, and off; compare all
/// three with DuckDB's `sql` (order-insensitive) and return the probe
/// scan's metrics when fully on. Asserts the exact key set never skips
/// fewer groups than min/max alone.
fn check(f: &Files, df: DataFrame, sql: &str) -> Result<ScanMetricsSnapshot> {
    Ok(check_both(f, df, sql)?.0)
}

/// `check`, returning (fully on, min/max only) metrics.
fn check_both(
    f: &Files,
    df: DataFrame,
    sql: &str,
) -> Result<(ScanMetricsSnapshot, ScanMetricsSnapshot)> {
    let plan = Optimizer::default().optimize(df.into_plan())?;
    let (on, m_on) = run(&plan, PlannerOptions::default())?;
    let (range, m_range) = run(&plan, RANGE)?;
    let (off, m_off) = run(&plan, OFF)?;
    let expected = duckdb_query(
        sql,
        &[],
        &[("facts", f.facts.as_path()), ("dims", f.dims.as_path())],
    )?;
    assert_batches_match(&on, &expected, false)?;
    assert_batches_match(&range, &expected, false)?;
    assert_batches_match(&off, &expected, false)?;
    assert_eq!(m_off.row_groups_pruned_join, 0);
    assert!(m_on.row_groups_pruned_join >= m_range.row_groups_pruned_join);
    Ok((m_on, m_range))
}

fn facts(f: &Files) -> DataFrame {
    scan_parquet_auto(&f.facts).unwrap()
}
fn dims(f: &Files) -> DataFrame {
    scan_parquet_auto(&f.dims).unwrap()
}
fn join(f: &Files, left: DataFrame, jt: JoinType) -> DataFrame {
    left.join(dims(f), vec![(col("k"), col("dk"))], jt).unwrap()
}

fn keys(r: std::ops::Range<i64>) -> Vec<Option<i64>> {
    r.map(Some).collect()
}

// ---- fires --------------------------------------------------------------

#[test]
fn inner_join_reads_only_groups_in_the_build_range() -> Result<()> {
    let f = files(keys(250..260));
    let m = check(
        &f,
        join(&f, facts(&f), JoinType::Inner),
        "SELECT * FROM facts JOIN dims ON k = dk",
    )?;
    assert_eq!(m.row_groups_pruned_join, 9);
    assert_eq!(m.row_groups_pruned, 0);
    Ok(())
}

#[test]
fn semi_join_prunes_too() -> Result<()> {
    let f = files(keys(250..260));
    let m = check(
        &f,
        join(&f, facts(&f), JoinType::Semi),
        "SELECT * FROM facts SEMI JOIN dims ON k = dk",
    )?;
    assert_eq!(m.row_groups_pruned_join, 9);
    Ok(())
}

#[test]
fn right_join_prunes_the_probe_and_keeps_unmatched_build_rows() -> Result<()> {
    // Build keys span [-5, 260]: by min/max, groups 0-2 can match and 3-9
    // can't. The exact keys (-5, 250..260) only hit group 2. -5 has no
    // partner and must still come out, NULL-padded.
    let mut d = keys(250..260);
    d.push(Some(-5));
    let f = files(d);
    let (m, range) = check_both(
        &f,
        join(&f, facts(&f), JoinType::Right),
        "SELECT * FROM facts RIGHT JOIN dims ON k = dk",
    )?;
    assert_eq!(range.row_groups_pruned_join, 7);
    assert_eq!(m.row_groups_pruned_join, 9);
    Ok(())
}

#[test]
fn prunes_through_a_probe_filter_and_projection() -> Result<()> {
    let f = files(keys(250..260));
    let left = facts(&f)
        .filter(col("v").neq(lit("v255")))?
        .project(vec![col("k"), col("v")])?;
    let m = check(
        &f,
        join(&f, left, JoinType::Inner),
        "SELECT * FROM (SELECT k, v FROM facts WHERE v <> 'v255') JOIN dims ON k = dk",
    )?;
    assert_eq!(m.row_groups_pruned_join, 9);
    Ok(())
}

#[test]
fn stacks_with_filter_pruning() -> Result<()> {
    // k >= 500 prunes groups 0-4 by stats; the build range [250, 659]
    // then rules out 7-9, and the exact keys rule out 5 as well. Each
    // group is counted once, under the first rule that removed it.
    let f = files(keys(250..260).into_iter().chain(keys(650..660)).collect());
    let left = facts(&f).filter(col("k").gt_eq(lit(500i64)))?;
    let (m, range) = check_both(
        &f,
        join(&f, left, JoinType::Inner),
        "SELECT * FROM facts JOIN dims ON k = dk WHERE k >= 500",
    )?;
    assert_eq!(m.row_groups_pruned, 5);
    assert_eq!(range.row_groups_pruned_join, 3);
    assert_eq!(m.row_groups_pruned_join, 4);
    Ok(())
}

#[test]
fn empty_build_side_skips_the_whole_probe() -> Result<()> {
    // The build filter removes every row: no probe row can match.
    let f = files(keys(250..260));
    for (jt, sql_join) in [
        (JoinType::Inner, "JOIN"),
        (JoinType::Semi, "SEMI JOIN"),
        (JoinType::Right, "RIGHT JOIN"),
    ] {
        let df = facts(&f).join(
            dims(&f).filter(col("dk").gt(lit(10_000i64)))?,
            vec![(col("k"), col("dk"))],
            jt,
        )?;
        let m = check(
            &f,
            df,
            &format!(
                "SELECT * FROM facts {sql_join} (SELECT * FROM dims WHERE dk > 10000) d ON k = d.dk"
            ),
        )?;
        assert_eq!(m.row_groups_pruned_join, 10, "{jt:?}");
        assert_eq!(m.rows_decoded, 0, "{jt:?}");
    }
    Ok(())
}

#[test]
fn all_null_build_keys_match_nothing() -> Result<()> {
    // NULL never equals anything, so the probe is skipped; the Right join
    // still returns both build rows, NULL-padded.
    let f = files(vec![None, None]);
    let m = check(
        &f,
        join(&f, facts(&f), JoinType::Inner),
        "SELECT * FROM facts JOIN dims ON k = dk",
    )?;
    assert_eq!(m.row_groups_pruned_join, 10);
    let m = check(
        &f,
        join(&f, facts(&f), JoinType::Right),
        "SELECT * FROM facts RIGHT JOIN dims ON k = dk",
    )?;
    assert_eq!(m.row_groups_pruned_join, 10);
    Ok(())
}

#[test]
fn scattered_keys_prune_only_with_the_exact_set() -> Result<()> {
    // Keys 99, 500, 995: min/max covers every group, but only groups 0, 5
    // and 9 hold a key. 99 is group 0's max and 500 is group 5's min, so
    // an off-by-one at either end of a group would show.
    let f = files(vec![Some(99), Some(500), Some(995)]);
    let (m, range) = check_both(
        &f,
        join(&f, facts(&f), JoinType::Inner),
        "SELECT * FROM facts JOIN dims ON k = dk",
    )?;
    assert_eq!(range.row_groups_pruned_join, 0);
    assert_eq!(m.row_groups_pruned_join, 7);
    Ok(())
}

#[test]
fn a_build_side_over_the_set_limit_falls_back_to_min_max() -> Result<()> {
    // 1025 distinct keys (one over the limit), all even, from 0 to 2048:
    // as a set they'd miss nothing useful, as a range they cover groups
    // 0-9 anyway, so both prune nothing; the point is the result.
    let f = files((0..1025).map(|i| Some(i * 2)).collect());
    let (m, range) = check_both(
        &f,
        join(&f, facts(&f), JoinType::Inner),
        "SELECT * FROM facts JOIN dims ON k = dk",
    )?;
    assert_eq!(m.row_groups_pruned_join, range.row_groups_pruned_join);
    Ok(())
}

// ---- must not fire ------------------------------------------------------

#[test]
fn must_not_prune_for_left_full_or_anti_joins() -> Result<()> {
    // Each keeps probe rows that have no match, so every group matters.
    let f = files(keys(250..260));
    for (jt, sql_join) in [
        (JoinType::Left, "LEFT JOIN"),
        (JoinType::Full, "FULL JOIN"),
        (JoinType::Anti, "ANTI JOIN"),
    ] {
        let m = check(
            &f,
            join(&f, facts(&f), jt),
            &format!("SELECT * FROM facts {sql_join} dims ON k = dk"),
        )?;
        assert_eq!(m.row_groups_pruned_join, 0, "{jt:?}");
        assert_eq!(m.rows_decoded, 1000, "{jt:?}");
    }
    Ok(())
}

#[test]
fn must_not_prune_for_left_full_or_anti_with_an_empty_build_side() -> Result<()> {
    // An empty build side is the extreme case: pruning would drop every
    // probe row these joins have to return.
    let f = files(keys(250..260));
    for (jt, sql_join) in [
        (JoinType::Left, "LEFT JOIN"),
        (JoinType::Full, "FULL JOIN"),
        (JoinType::Anti, "ANTI JOIN"),
    ] {
        let df = facts(&f).join(
            dims(&f).filter(col("dk").gt(lit(10_000i64)))?,
            vec![(col("k"), col("dk"))],
            jt,
        )?;
        let m = check(
            &f,
            df,
            &format!(
                "SELECT * FROM facts {sql_join} (SELECT * FROM dims WHERE dk > 10000) d ON k = d.dk"
            ),
        )?;
        assert_eq!(m.row_groups_pruned_join, 0, "{jt:?}");
    }
    Ok(())
}

#[test]
fn must_not_prune_below_a_probe_limit() -> Result<()> {
    // "Join the first 500 facts": pruning would change which 500 rows the
    // Limit keeps.
    let f = files(keys(250..260));
    let m = check(
        &f,
        join(&f, facts(&f).limit(500), JoinType::Inner),
        "SELECT * FROM (SELECT * FROM facts LIMIT 500) JOIN dims ON k = dk",
    )?;
    assert_eq!(m.row_groups_pruned_join, 0);
    Ok(())
}

#[test]
fn must_not_prune_when_a_projection_redefines_the_key() -> Result<()> {
    // The join key `k` is k + 500 from the file, so the file's stats for
    // k don't describe it.
    let f = files(keys(750..760));
    let left = facts(&f).project(vec![col("k").plus(lit(500i64)).alias("k"), col("v")])?;
    let m = check(
        &f,
        join(&f, left, JoinType::Inner),
        "SELECT * FROM (SELECT k + 500 AS k, v FROM facts) JOIN dims ON k = dk",
    )?;
    assert_eq!(m.row_groups_pruned_join, 0);
    Ok(())
}

// ---- TPC-H orders ⋈ lineitem ---------------------------------------------
//
// Release-only and skipped without data, like workloads_diff:
//     cargo test --release -p plume-bench --test join_pruning_diff

fn tpch_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/tpch")
}

/// lineitem ⋈ (orders WHERE <orders_filter>), counted and summed, on each
/// lineitem layout that exists. Returns (layout, probe metrics) pairs.
fn tpch_join(
    orders_filter: plume_core::Expr,
    orders_sql: &str,
) -> Result<Vec<(&'static str, ScanMetricsSnapshot)>> {
    let mut out = Vec::new();
    let orders = tpch_dir().join("orders.parquet");
    for layout in ["random", "sorted", "compound", "orderkey"] {
        let lineitem = tpch_dir().join(format!("lineitem_{layout}.parquet"));
        if !lineitem.exists() || !orders.exists() {
            eprintln!("SKIP tpch {layout}: data missing, run prepare_datasets");
            continue;
        }
        let df = scan_parquet_auto(&lineitem)?
            .join(
                scan_parquet_auto(&orders)?.filter(orders_filter.clone())?,
                vec![(col("l_orderkey"), col("o_orderkey"))],
                JoinType::Inner,
            )?
            .aggregate(
                vec![],
                vec![
                    plume_core::count_star().alias("n"),
                    plume_core::sum(col("l_extendedprice")).alias("revenue"),
                ],
            )?;
        let plan = Optimizer::default().optimize(df.into_plan())?;
        let (on, m_on) = run(&plan, PlannerOptions::default())?;
        let (off, m_off) = run(&plan, OFF)?;
        let expected = duckdb_query(
            &format!(
                "SELECT count(*) AS n, sum(l_extendedprice) AS revenue \
                 FROM lineitem JOIN orders ON l_orderkey = o_orderkey WHERE {orders_sql}"
            ),
            &[],
            &[
                ("lineitem", lineitem.as_path()),
                ("orders", orders.as_path()),
            ],
        )?;
        assert_batches_match(&on, &expected, false)?;
        assert_batches_match(&off, &expected, false)?;
        assert_eq!(m_off.row_groups_pruned_join, 0);
        eprintln!(
            "tpch {layout}: {}/{} lineitem row groups pruned by the join",
            m_on.row_groups_pruned_join, m_on.row_groups_total
        );
        out.push((layout, m_on));
    }
    Ok(out)
}

#[test]
fn tpch_join_with_an_order_date_filter_matches_duckdb() -> Result<()> {
    if cfg!(debug_assertions) {
        eprintln!("SKIP: debug build, run with --release");
        return Ok(());
    }
    // One week of orders. Order dates are independent of order keys in
    // dbgen, so the build keys span nearly the whole key range and little
    // can be pruned on any layout; this checks correctness.
    tpch_join(
        col("o_orderdate")
            .gt_eq(plume_core::lit_date("1995-03-01"))
            .and(col("o_orderdate").lt(plume_core::lit_date("1995-03-08"))),
        "o_orderdate >= DATE '1995-03-01' AND o_orderdate < DATE '1995-03-08'",
    )?;
    Ok(())
}

#[test]
fn tpch_join_on_a_key_range_prunes_the_orderkey_layout() -> Result<()> {
    if cfg!(debug_assertions) {
        eprintln!("SKIP: debug build, run with --release");
        return Ok(());
    }
    // The first 1% of order keys. lineitem sorted by l_orderkey keeps
    // those rows in the first row group or two.
    let results = tpch_join(col("o_orderkey").lt(lit(60_000i64)), "o_orderkey < 60000")?;
    if let Some((_, m)) = results.iter().find(|(l, _)| *l == "orderkey") {
        assert!(
            m.row_groups_pruned_join * 10 >= m.row_groups_total * 9,
            "expected most groups pruned: {m:?}"
        );
    }
    Ok(())
}

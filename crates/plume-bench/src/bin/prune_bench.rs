//! Measure how much row-group pruning skips on the TPC-H and taxi data.
//!
//! For every dataset × layout × query in `workloads`, it runs:
//!   - Plume with pruning off: the full optimizer, then the scans'
//!     pushdown predicates cleared. Filters still move and columns are
//!     still pruned, so the only difference from "on" is that the scan
//!     reads every row group.
//!   - Plume with pruning on: the full optimizer, unchanged.
//!   - DuckDB on the same files, as a reference time.
//!
//! Each gets one warm-up run, then the median wall time of `--runs` runs
//! (default 5). Plume's result must equal DuckDB's in both modes or the
//! run stops.
//!
//! Scan metrics come from the first Parquet scan in the plan, which is
//! always the fact table (`lineitem` / `trips`); in the join query
//! `orders` is the second scan and is not counted. Actual selectivity is
//! the sum of the query's `n` column over the fact table's row count. For
//! the join that is still the fraction of lineitem rows kept, because
//! each lineitem row matches exactly one order.
//!
//! Writes results/phase1/pruning.csv and results/phase1/env.json.
//!
//! With `--phase 2` it measures all four techniques instead: every query
//! in `workloads::phase2_for_dataset` on every layout of
//! `workloads::layouts`, each run with its own technique on and off and
//! every other technique left on:
//!   - filter: off = pushdown predicates cleared (as in Phase 1)
//!   - limit:  off = the optimizer without the LimitPruning rule
//!   - topk:   off = `PlannerOptions::top_k_pruning = false`
//!   - join:   off = `PlannerOptions::join_pruning = false`
//!
//! Two techniques get a third mode, for the optional extras:
//!   - join:   range = join pruning with min/max only, no exact key sets
//!   - topk:   noinit = top-k pruning without the stats-initialized boundary
//!
//! Results are checked with `Workload::verify` in both modes. Each
//! technique's own skip counter is recorded next to the general scan
//! metrics, in results/phase2/techniques.csv (plus env.json).
//!
//!     cargo run --release -p plume-bench --bin prune_bench
//!     cargo run --release -p plume-bench --bin prune_bench -- --phase 2
//!
//! Options:
//!     --phase 1|2         which measurement to run (default 1)
//!     --runs <n>          timed runs per measurement (default 5)
//!     --dataset tpch|taxi only run one dataset
//!     --query <substr>    only run queries whose name contains this

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::RecordBatch;
use duckdb::Connection;
use parquet::file::reader::{FileReader, SerializedFileReader};
use plume_bench::harness::assert_batches_match;
use plume_bench::workloads::{self, Dataset, Tables, Technique, Workload, LAYOUTS};
use plume_core::LogicalPlan;
use plume_exec::{
    create_physical_plan_with_metrics, create_physical_plan_with_options, EvalResult,
    PlannerOptions, ScanMetricsSnapshot,
};
use plume_optimizer::{
    ColumnPruning, ConstantFolding, DeadProjectionElim, LimitPushdown, Optimizer,
    PredicatePushdown, SortLimitToTopK,
};
use serde_json::json;

const CSV_HEADER: &str = "dataset,layout,query,target_selectivity,actual_selectivity,pruning,\
row_groups_total,row_groups_pruned,bytes_total,bytes_read,rows_decoded,rows_in_file,\
plume_ms,duckdb_ms";

const PHASE2_HEADER: &str = "dataset,layout,query,technique,target_selectivity,\
actual_selectivity,mode,row_groups_total,row_groups_pruned,row_groups_pruned_limit,\
row_groups_pruned_topk,row_groups_pruned_join,row_groups_read,bytes_total,bytes_read,\
rows_decoded,rows_in_file,plume_ms,duckdb_ms";

struct Args {
    phase: u8,
    runs: usize,
    datasets: Vec<Dataset>,
    query: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        phase: 1,
        runs: 5,
        datasets: vec![Dataset::Tpch, Dataset::Taxi],
        query: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--runs" => {
                let v = it.next().context("--runs needs a value")?;
                args.runs = v.parse().with_context(|| format!("bad --runs {v}"))?;
                ensure!(args.runs > 0, "--runs must be at least 1");
            }
            "--dataset" => match it.next().as_deref() {
                Some("tpch") => args.datasets = vec![Dataset::Tpch],
                Some("taxi") => args.datasets = vec![Dataset::Taxi],
                other => bail!("--dataset expects tpch or taxi, got {other:?}"),
            },
            "--query" => args.query = Some(it.next().context("--query needs a value")?),
            "--phase" => match it.next().as_deref() {
                Some("1") => args.phase = 1,
                Some("2") => args.phase = 2,
                other => bail!("--phase expects 1 or 2, got {other:?}"),
            },
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(args)
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.phase == 2 {
        return phase2(&args);
    }
    let root = workspace_root();
    let data = root.join("data");
    let out_dir = root.join("results").join("phase1");
    fs::create_dir_all(&out_dir)?;

    let mut csv = String::from(CSV_HEADER);
    csv.push('\n');

    for &dataset in &args.datasets {
        let queries: Vec<Workload> = workloads::for_dataset(dataset)
            .into_iter()
            .filter(|w| args.query.as_ref().map_or(true, |q| w.name.contains(q)))
            .collect();
        for layout in LAYOUTS {
            let tables = Tables::new(&data, dataset, layout);
            ensure!(
                tables.exist(),
                "{} {layout} data missing; run \
                 `cargo run --release -p plume-bench --bin prepare_datasets`",
                dataset.name()
            );
            let fact = match dataset {
                Dataset::Tpch => "lineitem",
                Dataset::Taxi => "trips",
            };
            let rows_in_file = parquet_rows(tables.path(fact)?)?;
            let duck = duckdb_connection(&tables)?;
            for q in &queries {
                for line in measure(q, &tables, &duck, rows_in_file, args.runs)? {
                    csv.push_str(&line);
                    csv.push('\n');
                }
            }
        }
    }

    let csv_path = out_dir.join("pruning.csv");
    fs::write(&csv_path, csv)?;
    let env_path = out_dir.join("env.json");
    fs::write(
        &env_path,
        serde_json::to_string_pretty(&env_info(&args)?)? + "\n",
    )?;
    println!("wrote {} and {}", csv_path.display(), env_path.display());
    Ok(())
}

fn phase2(args: &Args) -> Result<()> {
    let root = workspace_root();
    let data = root.join("data");
    let out_dir = root.join("results").join("phase2");
    fs::create_dir_all(&out_dir)?;

    // Recorded before the run starts, so the commit and dirty flag
    // describe the code that was measured.
    let env = env_info(args)?;
    let mut csv = String::from(PHASE2_HEADER);
    csv.push('\n');
    for &dataset in &args.datasets {
        let queries: Vec<Workload> = workloads::phase2_for_dataset(dataset)
            .into_iter()
            .filter(|w| args.query.as_ref().map_or(true, |q| w.name.contains(q)))
            .collect();
        for layout in workloads::layouts(dataset) {
            let tables = Tables::new(&data, dataset, layout);
            ensure!(
                tables.exist(),
                "{} {layout} data missing; run \
                 `cargo run --release -p plume-bench --bin prepare_datasets`",
                dataset.name()
            );
            let fact = match dataset {
                Dataset::Tpch => "lineitem",
                Dataset::Taxi => "trips",
            };
            let rows_in_file = parquet_rows(tables.path(fact)?)?;
            let duck = duckdb_connection(&tables)?;
            for q in &queries {
                for line in measure_technique(q, &tables, &duck, rows_in_file, args.runs)? {
                    csv.push_str(&line);
                    csv.push('\n');
                }
            }
        }
    }
    let csv_path = out_dir.join("techniques.csv");
    fs::write(&csv_path, csv)?;
    let env_path = out_dir.join("env.json");
    fs::write(&env_path, serde_json::to_string_pretty(&env)? + "\n")?;
    println!("wrote {} and {}", csv_path.display(), env_path.display());
    Ok(())
}

/// The default optimizer without LimitPruning: the "off" plan for LIMIT
/// queries.
fn without_limit_pruning() -> Optimizer {
    Optimizer::with_rules(vec![
        Box::new(ConstantFolding),
        Box::new(DeadProjectionElim),
        Box::new(PredicatePushdown),
        Box::new(ColumnPruning),
        Box::new(LimitPushdown),
        Box::new(SortLimitToTopK),
    ])
}

/// Phase 2: run one query with its technique off and on, verify both, and
/// return the CSV lines.
fn measure_technique(
    q: &Workload,
    tables: &Tables,
    duck: &Connection,
    rows_in_file: u64,
    runs: usize,
) -> Result<Vec<String>> {
    let ctx = format!("{} {} {}", tables.dataset.name(), tables.layout, q.name);
    let raw = q.plan(tables)?.into_plan();
    let on_plan = Optimizer::default().optimize(raw.clone())?;
    let on_opts = PlannerOptions::default();
    let (off_plan, off_opts) = match q.technique {
        Technique::Filter => (strip_pushdown(on_plan.clone()), on_opts),
        Technique::Limit => (without_limit_pruning().optimize(raw)?, on_opts),
        Technique::TopK => (
            on_plan.clone(),
            PlannerOptions {
                top_k_pruning: false,
                ..on_opts
            },
        ),
        Technique::Join => (
            on_plan.clone(),
            PlannerOptions {
                join_pruning: false,
                ..on_opts
            },
        ),
    };

    let (duck_time, expected) = time_runs(runs, || duckdb_run(duck, &q.sql))?;

    let mut modes = vec![("off", &off_plan, off_opts)];
    match q.technique {
        Technique::Join => modes.push((
            "range",
            &on_plan,
            PlannerOptions {
                join_key_set: false,
                ..on_opts
            },
        )),
        Technique::TopK => modes.push((
            "noinit",
            &on_plan,
            PlannerOptions {
                top_k_stats_init: false,
                ..on_opts
            },
        )),
        _ => {}
    }
    modes.push(("on", &on_plan, on_opts));

    let mut lines = Vec::new();
    for (mode, plan, opts) in modes {
        let (plume_time, (batches, metrics)) = time_runs(runs, || plume_run_with(plan, opts))?;
        q.verify(&batches, &expected, duck)
            .with_context(|| format!("{ctx} {} {mode}: wrong result", q.technique.name()))?;
        let m = metrics
            .first()
            .copied()
            .with_context(|| format!("{ctx}: plan has no Parquet scan"))?;
        let selectivity = sum_n(&batches)
            .ok()
            .map(|n| format!("{:.6}", n as f64 / rows_in_file as f64))
            .unwrap_or_default();
        let mut line = String::new();
        write!(
            line,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3}",
            tables.dataset.name(),
            tables.layout,
            q.name,
            q.technique.name(),
            q.target_selectivity
                .map(|t| t.to_string())
                .unwrap_or_default(),
            selectivity,
            mode,
            m.row_groups_total,
            m.row_groups_pruned,
            m.row_groups_pruned_limit,
            m.row_groups_pruned_topk,
            m.row_groups_pruned_join,
            m.row_groups_read,
            m.bytes_total,
            m.bytes_read,
            m.rows_decoded,
            rows_in_file,
            ms(plume_time),
            ms(duck_time),
        )?;
        println!(
            "{ctx:<44} {:<6} {mode:<6}  read {:>3}/{:<3} groups  plume {:>9.1} ms  duckdb {:>8.1} ms",
            q.technique.name(),
            m.row_groups_read,
            m.row_groups_total,
            ms(plume_time),
            ms(duck_time)
        );
        lines.push(line);
    }
    Ok(lines)
}

/// Run one query in every mode and return its CSV lines.
fn measure(
    q: &Workload,
    tables: &Tables,
    duck: &Connection,
    rows_in_file: u64,
    runs: usize,
) -> Result<Vec<String>> {
    let ctx = format!("{} {} {}", tables.dataset.name(), tables.layout, q.name);
    let optimized = Optimizer::default().optimize(q.plan(tables)?.into_plan())?;
    let unpruned = strip_pushdown(optimized.clone());

    let (duck_time, expected) = time_runs(runs, || duckdb_run(duck, &q.sql))?;

    let mut lines = Vec::new();
    for (mode, plan) in [("off", &unpruned), ("on", &optimized)] {
        let (plume_time, (batches, metrics)) = time_runs(runs, || plume_run(plan))?;
        assert_batches_match(&batches, &expected, false)
            .with_context(|| format!("{ctx} pruning {mode}: Plume and DuckDB disagree"))?;
        let m = metrics
            .first()
            .copied()
            .with_context(|| format!("{ctx}: plan has no Parquet scan"))?;
        let matched = sum_n(&batches).with_context(|| ctx.clone())?;
        let mut line = String::new();
        write!(
            line,
            "{},{},{},{},{:.6},{},{},{},{},{},{},{},{:.3},{:.3}",
            tables.dataset.name(),
            tables.layout,
            q.name,
            q.target_selectivity
                .map(|t| t.to_string())
                .unwrap_or_default(),
            matched as f64 / rows_in_file as f64,
            mode,
            m.row_groups_total,
            m.row_groups_pruned,
            m.bytes_total,
            m.bytes_read,
            m.rows_decoded,
            rows_in_file,
            ms(plume_time),
            ms(duck_time),
        )?;
        println!(
            "{ctx:<40} pruning {mode:<3}  {}/{} row groups pruned  plume {:>9.1} ms  duckdb {:>8.1} ms",
            m.row_groups_pruned,
            m.row_groups_total,
            ms(plume_time),
            ms(duck_time)
        );
        lines.push(line);
    }
    Ok(lines)
}

/// One warm-up call, then `runs` timed calls. Returns the median time and
/// the last call's output.
fn time_runs<T>(runs: usize, mut f: impl FnMut() -> Result<T>) -> Result<(Duration, T)> {
    f()?;
    let mut times = Vec::with_capacity(runs);
    let mut last = None;
    for _ in 0..runs {
        let start = Instant::now();
        let out = f()?;
        times.push(start.elapsed());
        last = Some(out);
    }
    times.sort();
    Ok((times[runs / 2], last.expect("runs > 0")))
}

fn plume_run_with(
    plan: &LogicalPlan,
    opts: PlannerOptions,
) -> Result<(Vec<RecordBatch>, Vec<ScanMetricsSnapshot>)> {
    let (exec, handles) = create_physical_plan_with_options(plan, opts)?;
    let batches = exec.execute()?.collect::<EvalResult<Vec<_>>>()?;
    Ok((batches, handles.iter().map(|h| h.snapshot()).collect()))
}

fn plume_run(plan: &LogicalPlan) -> Result<(Vec<RecordBatch>, Vec<ScanMetricsSnapshot>)> {
    let (exec, handles) = create_physical_plan_with_metrics(plan)?;
    let batches = exec.execute()?.collect::<EvalResult<Vec<_>>>()?;
    Ok((batches, handles.iter().map(|h| h.snapshot()).collect()))
}

fn duckdb_connection(tables: &Tables) -> Result<Connection> {
    let conn = Connection::open_in_memory()?;
    for (name, path) in tables.duckdb_views() {
        let p = path.to_str().context("non-UTF-8 path")?;
        conn.execute_batch(&format!(
            "CREATE VIEW {name} AS SELECT * FROM read_parquet('{p}');"
        ))?;
    }
    Ok(conn)
}

fn duckdb_run(conn: &Connection, sql: &str) -> Result<Vec<RecordBatch>> {
    let mut stmt = conn.prepare(sql)?;
    Ok(stmt.query_arrow([])?.collect())
}

/// Clear every scan's pushdown predicates, which turns off row-group
/// pruning and leaves the rest of the optimized plan as it was.
fn strip_pushdown(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Scan {
            source,
            schema,
            projection,
            limit,
            ..
        } => LogicalPlan::Scan {
            source,
            schema,
            projection,
            pushdown_predicates: Vec::new(),
            limit,
            // The hint is only safe while the pushdown predicates cover
            // the Filter above, so it has to go with them.
            limit_hint: None,
        },
        LogicalPlan::Filter {
            input,
            predicate,
            schema,
        } => LogicalPlan::Filter {
            input: Box::new(strip_pushdown(*input)),
            predicate,
            schema,
        },
        LogicalPlan::Projection {
            input,
            exprs,
            schema,
        } => LogicalPlan::Projection {
            input: Box::new(strip_pushdown(*input)),
            exprs,
            schema,
        },
        LogicalPlan::Aggregate {
            input,
            group_exprs,
            agg_exprs,
            schema,
        } => LogicalPlan::Aggregate {
            input: Box::new(strip_pushdown(*input)),
            group_exprs,
            agg_exprs,
            schema,
        },
        LogicalPlan::Limit { input, n, schema } => LogicalPlan::Limit {
            input: Box::new(strip_pushdown(*input)),
            n,
            schema,
        },
        LogicalPlan::Sort {
            input,
            exprs,
            schema,
        } => LogicalPlan::Sort {
            input: Box::new(strip_pushdown(*input)),
            exprs,
            schema,
        },
        LogicalPlan::TopK {
            input,
            exprs,
            k,
            schema,
        } => LogicalPlan::TopK {
            input: Box::new(strip_pushdown(*input)),
            exprs,
            k,
            schema,
        },
        LogicalPlan::Join {
            left,
            right,
            on,
            filter,
            join_type,
            schema,
        } => LogicalPlan::Join {
            left: Box::new(strip_pushdown(*left)),
            right: Box::new(strip_pushdown(*right)),
            on,
            filter,
            join_type,
            schema,
        },
    }
}

/// Sum of the `n` (count) column across all result rows.
fn sum_n(batches: &[RecordBatch]) -> Result<u64> {
    let mut total = 0i64;
    for b in batches {
        let col = b
            .column_by_name("n")
            .context("query result has no `n` column")?;
        let arr = col
            .as_primitive_opt::<Int64Type>()
            .context("`n` is not Int64")?;
        total += arr.iter().flatten().sum::<i64>();
    }
    Ok(total as u64)
}

fn parquet_rows(path: &Path) -> Result<u64> {
    let reader = SerializedFileReader::new(fs::File::open(path)?)?;
    Ok(reader.metadata().file_metadata().num_rows() as u64)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn command_output(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn cpu_model() -> String {
    command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(|| {
            fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|v| v.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unknown".into())
}

fn env_info(args: &Args) -> Result<serde_json::Value> {
    let root = workspace_root();
    let root_s = root.to_str().unwrap_or(".");
    let sha = command_output("git", &["-C", root_s, "rev-parse", "HEAD"]);
    let dirty = command_output(
        "git",
        &[
            "-C",
            root_s,
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
    )
    .map(|s| !s.is_empty());
    let duck = Connection::open_in_memory()?;
    let duckdb_version: String = duck.query_row("SELECT version()", [], |r| r.get(0))?;
    let duckdb_threads: i64 =
        duck.query_row("SELECT current_setting('threads')", [], |r| r.get(0))?;
    Ok(json!({
        "command": if args.phase == 2 {
            "cargo run --release -p plume-bench --bin prune_bench -- --phase 2"
        } else {
            "cargo run --release -p plume-bench --bin prune_bench"
        },
        "date_utc": command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "git_commit": sha,
        "git_dirty": dirty,
        "rustc": command_output("rustc", &["--version"]),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu": cpu_model(),
        "cores": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "duckdb_version": duckdb_version,
        "duckdb_threads": duckdb_threads,
        "plume_threads": 1,
        "warmup_runs": 1,
        "timed_runs": args.runs,
        "timing": "median wall time",
    }))
}

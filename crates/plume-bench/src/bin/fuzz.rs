//! Run randomized differential tests (see `plume_bench::random_plans`,
//! and `plume_bench::random_real` for `--real`).
//!
//!     cargo run --release -p plume-bench --bin fuzz -- --cases 10000
//!     cargo run -p plume-bench --bin fuzz -- --seed 1234      # replay one
//!     cargo run --release -p plume-bench --bin fuzz -- --real tpch --cases 200
//!
//! Options:
//!     --seed <s>     first seed (default 0)
//!     --cases <n>    how many consecutive seeds to run (default 1000;
//!                    1 when --seed is given)
//!     --keep-going   don't stop at the first failure
//!     --no-shrink    report failures as generated, without shrinking
//!     --save <dir>   write each shrunk failure (tables and SQL) to
//!                    <dir>/seed-<s>/
//!     --real <ds>    query the real dataset `tpch` or `taxi` (needs
//!                    `prepare_datasets`) instead of generated tables
//!     --data-dir <d> where the real datasets are (default data)

use anyhow::{bail, Context, Result};
use std::path::PathBuf;

use plume_bench::random_plans;
use plume_bench::random_real::{self, RealSource};
use plume_bench::workloads::Dataset;

fn main() -> Result<()> {
    let mut seed = 0u64;
    let mut cases: Option<u64> = None;
    let mut seed_given = false;
    let mut keep_going = false;
    let mut shrink = true;
    let mut save: Option<PathBuf> = None;
    let mut real: Option<Dataset> = None;
    let mut data_dir = PathBuf::from("data");
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seed" => {
                let v = it.next().context("--seed needs a value")?;
                seed = v.parse().with_context(|| format!("bad --seed {v}"))?;
                seed_given = true;
            }
            "--cases" => {
                let v = it.next().context("--cases needs a value")?;
                cases = Some(v.parse().with_context(|| format!("bad --cases {v}"))?);
            }
            "--keep-going" => keep_going = true,
            "--no-shrink" => shrink = false,
            "--save" => save = Some(it.next().context("--save needs a value")?.into()),
            "--real" => {
                real = Some(match it.next().as_deref() {
                    Some("tpch") => Dataset::Tpch,
                    Some("taxi") => Dataset::Taxi,
                    v => bail!("--real takes tpch or taxi, not {v:?}"),
                })
            }
            "--data-dir" => data_dir = it.next().context("--data-dir needs a value")?.into(),
            other => bail!("unknown argument {other}"),
        }
    }
    let cases = cases.unwrap_or(if seed_given { 1 } else { 1000 });
    let sources: Vec<RealSource> = match real {
        Some(ds) => {
            eprintln!("sampling {} columns...", ds.name());
            RealSource::load_all(&data_dir, ds)?
        }
        None => Vec::new(),
    };
    let generate = |s: u64| {
        if sources.is_empty() {
            random_plans::generate(s)
        } else {
            random_real::generate(&sources, s)
        }
    };
    let mut failed = Vec::new();
    let mut counts = [0u64; 10];
    // PLUME_FUZZ_TRACE=1 prints each seed before running it, to find a
    // case that aborts the process (e.g. a DuckDB assertion).
    let trace = std::env::var_os("PLUME_FUZZ_TRACE").is_some();
    for s in seed..seed + cases {
        if trace {
            eprintln!("seed {s}");
        }
        let outcome = generate(s).and_then(|c| {
            if seed_given && cases == 1 {
                if !sources.is_empty() {
                    let src = random_real::source_for(&sources, s);
                    println!("layout: {} {}", src.dataset.name(), src.layout);
                }
                println!("SQL: {}", c.sql);
            }
            c.run()
        });
        if let Ok(c) = &outcome {
            let flags = [
                c.join,
                c.aggregate,
                c.top_k,
                c.sort,
                c.limit,
                c.filter_pruned,
                c.filter_skipped,
                c.limit_pruned,
                c.top_k_pruned,
                c.join_pruned,
            ];
            for (n, f) in counts.iter_mut().zip(flags) {
                *n += f as u64;
            }
        }
        if let Err(e) = outcome {
            eprintln!("FAIL seed {s}: {e:#}\n");
            if shrink {
                let budget = random_plans::SHRINK_BUDGET;
                let shrunk = if sources.is_empty() {
                    random_plans::shrink(s, budget)
                } else {
                    random_real::shrink(&sources, s, budget)
                };
                match shrunk {
                    Ok(Some(shrunk)) => {
                        eprintln!("{}\n", shrunk.report());
                        if let Some(dir) = &save {
                            let dir = dir.join(format!("seed-{s}"));
                            shrunk.save(&dir)?;
                            eprintln!("saved to {}\n", dir.display());
                        }
                    }
                    Ok(None) => eprintln!("(not shrunk: the failure is outside Plume)\n"),
                    Err(e) => eprintln!("(not shrunk: {e:#})\n"),
                }
            }
            failed.push(s);
            if !keep_going {
                break;
            }
        }
    }
    let names = [
        "join",
        "aggregate",
        "TopK",
        "Sort",
        "Limit",
        "filter pruning skipped groups",
        "filter skipped on full-match groups",
        "LIMIT pruning skipped groups",
        "top-k pruning skipped groups",
        "join pruning skipped groups",
    ];
    println!("coverage (cases whose optimized plan has / did each):");
    for (n, c) in names.iter().zip(counts) {
        println!("  {n:<38} {c}");
    }
    if failed.is_empty() {
        println!("{cases} cases passed (seeds {seed}..{})", seed + cases);
        Ok(())
    } else {
        bail!("{} failing seed(s): {failed:?}", failed.len())
    }
}

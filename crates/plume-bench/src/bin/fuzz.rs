//! Run randomized differential tests (see `plume_bench::random_plans`).
//!
//!     cargo run --release -p plume-bench --bin fuzz -- --cases 10000
//!     cargo run -p plume-bench --bin fuzz -- --seed 1234      # replay one
//!
//! Options:
//!     --seed <s>     first seed (default 0)
//!     --cases <n>    how many consecutive seeds to run (default 1000;
//!                    1 when --seed is given)
//!     --keep-going   don't stop at the first failure
//!     --no-shrink    report failures as generated, without shrinking
//!     --save <dir>   write each shrunk failure (tables and SQL) to
//!                    <dir>/seed-<s>/

use anyhow::{bail, Context, Result};
use std::path::PathBuf;

use plume_bench::random_plans;

fn main() -> Result<()> {
    let mut seed = 0u64;
    let mut cases: Option<u64> = None;
    let mut seed_given = false;
    let mut keep_going = false;
    let mut shrink = true;
    let mut save: Option<PathBuf> = None;
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
            other => bail!("unknown argument {other}"),
        }
    }
    let cases = cases.unwrap_or(if seed_given { 1 } else { 1000 });
    let mut failed = Vec::new();
    let mut counts = [0u64; 10];
    // PLUME_FUZZ_TRACE=1 prints each seed before running it, to find a
    // case that aborts the process (e.g. a DuckDB assertion).
    let trace = std::env::var_os("PLUME_FUZZ_TRACE").is_some();
    for s in seed..seed + cases {
        if trace {
            eprintln!("seed {s}");
        }
        let outcome = random_plans::generate(s).and_then(|c| {
            if seed_given && cases == 1 {
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
                match random_plans::shrink(s, random_plans::SHRINK_BUDGET) {
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

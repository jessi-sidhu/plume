//! Randomized differential tests against DuckDB, a fixed range of seeds
//! so CI is reproducible. More seeds: PLUME_RANDOM_CASES=5000, or the
//! `fuzz` binary. A failure names its seed and prints a shrunk version of
//! the case; replay the full one with
//! `cargo run -p plume-bench --bin fuzz -- --seed <s>`.

use plume_bench::random_plans;

#[test]
fn random_plans_match_duckdb() {
    let cases: u64 = std::env::var("PLUME_RANDOM_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let mut failures = Vec::new();
    for seed in 0..cases {
        if let Err(e) = random_plans::generate(seed).and_then(|c| c.run().map(|_| ())) {
            failures.push((seed, format!("{e:#}")));
        }
    }
    if let Some((seed, first)) = failures.first() {
        // A small budget: this is a debug build, and the fuzz binary can
        // shrink further.
        let shrunk = match random_plans::shrink(*seed, 300) {
            Ok(Some(s)) => s.report(),
            Ok(None) => "(not shrunk: the failure is outside Plume)".into(),
            Err(e) => format!("(not shrunk: {e:#})"),
        };
        panic!(
            "{} of {cases} random plans failed; first:\n{first}\n\n{shrunk}",
            failures.len()
        );
    }
}

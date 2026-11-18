//! Randomized differential tests against DuckDB, a fixed range of seeds
//! so CI is reproducible. More seeds: PLUME_RANDOM_CASES=5000, or the
//! `fuzz` binary. A failure names its seed; replay it with
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
            failures.push(format!("{e:#}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {cases} random plans failed; first:\n{}",
        failures.len(),
        failures[0]
    );
}

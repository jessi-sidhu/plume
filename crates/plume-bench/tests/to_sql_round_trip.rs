//! The plan-to-SQL renderer over random plans: each case's plan, rendered
//! back to SQL, must give the same DuckDB result as the case's
//! hand-built SQL (checked the same way Plume's result is, so LIMIT
//! without ORDER BY and ties in a top-k are allowed to differ). The other
//! diff tests check the renderer against their own hand-written SQL.
//! More seeds: PLUME_RANDOM_CASES=5000.

use plume_bench::random_plans;

#[test]
fn rendered_random_plans_match_their_sql() {
    let cases: u64 = std::env::var("PLUME_RANDOM_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let failures: Vec<(u64, String)> = (0..cases)
        .filter_map(|seed| {
            random_plans::generate(seed)
                .and_then(|c| c.check_rendered())
                .err()
                .map(|e| (seed, format!("{e:#}")))
        })
        .collect();
    if let Some((_, first)) = failures.first() {
        panic!(
            "{} of {cases} rendered plans disagreed; first:\n{first}",
            failures.len()
        );
    }
}

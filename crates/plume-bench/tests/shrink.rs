//! The shrinker for randomized test cases (`random_plans::shrink`).

use std::path::Path;

use anyhow::{anyhow, Result};
use plume_bench::harness::reference_connection;
use plume_bench::random_plans::{
    generate, generate_with, shrink_by, Case, Failure, FailureKind, Outcome, Shrink,
};

/// Rows of the Parquet file at `path` with `a_k = 3`.
fn rows_with_k3(path: &Path) -> Result<usize> {
    let conn = reference_connection()?;
    let n: usize = conn.query_row(
        &format!(
            "SELECT count(*) FROM read_parquet('{}') WHERE a_k = 3",
            path.display()
        ),
        [],
        |r| r.get(0),
    )?;
    Ok(n)
}

fn fake_failure() -> Failure {
    Failure {
        mode: "fake",
        kind: FailureKind::WrongResult,
        error: anyhow!("t1 has two rows with a_k = 3"),
    }
}

#[test]
fn shrinks_to_the_rows_and_steps_the_failure_needs() -> Result<()> {
    // A fake bug: "fails" whenever t1 holds at least two rows with
    // a_k = 3. The minimal case keeps exactly two t1 rows, no t2 rows,
    // and none of the query's droppable steps.
    let mut fails = |c: &Case| -> Option<Failure> {
        (rows_with_k3(&c.tables()[0].1).ok()? >= 2).then(fake_failure)
    };
    let (seed, case) = (0..200u64)
        .map(|s| (s, generate(s).unwrap()))
        .find(|(_, c)| fails(c).is_some() && !c.steps.is_empty() && c.full_rows.1 > 0)
        .expect("some seed has two a_k = 3 rows, steps, and a non-empty t2");
    let regen = |s: &Shrink| generate_with(seed, s);
    let shrunk = shrink_by(case, fake_failure(), 5000, &regen, &mut fails);
    assert_eq!(shrunk.case.seed, seed);
    assert_eq!(shrunk.shrink.t1_rows.as_ref().map(Vec::len), Some(2));
    assert_eq!(shrunk.shrink.t2_rows.as_deref(), Some(&[][..]));
    assert!(shrunk.case.steps.is_empty(), "{:?}", shrunk.case.steps);
    assert_eq!(rows_with_k3(&shrunk.case.tables()[0].1)?, 2);
    Ok(())
}

#[test]
fn a_passing_case_is_left_alone() -> Result<()> {
    let mut never = |_: &Case| -> Option<Failure> { None };
    let regen = |s: &Shrink| generate_with(1, s);
    let shrunk = shrink_by(generate(1)?, fake_failure(), 100, &regen, &mut never);
    assert!(shrunk.shrink.skip.is_empty());
    assert!(shrunk.shrink.t1_rows.is_none() && shrunk.shrink.t2_rows.is_none());
    Ok(())
}

#[test]
fn shrunk_cases_are_still_valid_queries() -> Result<()> {
    // Every candidate the shrinker can build must be a valid query that
    // Plume and DuckDB agree on; otherwise shrinking a real failure could
    // wander off to a bogus one. Drop each step alone, and every other row.
    for seed in 0..30u64 {
        let case = generate(seed)?;
        let mut cands = Vec::new();
        for &step in &case.steps {
            let mut s = Shrink::default();
            s.skip.insert(step);
            cands.push(s);
        }
        let (n1, n2) = case.full_rows;
        cands.push(Shrink {
            t1_rows: Some((0..n1).step_by(2).collect()),
            t2_rows: Some((1..n2).step_by(2).collect()),
            ..Shrink::default()
        });
        for cand in cands {
            let c = generate_with(seed, &cand)?;
            if let Outcome::Fail(f) = c.outcome()? {
                panic!("seed {seed} cut by {cand:?} fails: {:#}", f.error);
            }
        }
    }
    Ok(())
}

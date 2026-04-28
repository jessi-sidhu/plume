# plume

A small, programmatic-API query engine in Rust. No SQL parser; the plan **is**
the API. Built to focus on execution quality (vectorized eval, Parquet-aware
scans, an honest optimizer) rather than SQL surface area.

Status: Week 1 — scaffolding, fixture generation, DuckDB diff harness.

## Locked scope decisions

- **Lazy evaluation only.** Plans are built, then `.collect()` executes.
- **Arrow as the only type system.** No parallel type hierarchy.
- **Owned data, not zero-copy.** Optimize allocations later, with profiler evidence.
- **Single-threaded first.** Parallelism is a stretch goal (Week 13).
- **Correctness over performance.** Every query is diffed against DuckDB.

## Layout

```
crates/
  plume-core/        LogicalPlan, Expr, Schema  (Week 2+)
  plume-exec/        ExecutionPlan, operators, evaluator  (Week 3+)
  plume-optimizer/   OptimizerRule + rules  (Week 8+)
  plume-io/          CSV + Parquet scans  (Week 4 / Week 7)
  plume-bench/       DuckDB diff harness, fixture generator, benches
fixtures/            (gitignored) generated test data
docs/                architecture.md, profiling.md (W11), benchmarks.md (W12)
```

## Quickstart

```sh
# 1. Build everything (first build pulls bundled DuckDB; expect 3–5 min cold).
cargo build --workspace

# 2. Generate the Week-1 fixtures into ./fixtures/.
cargo run -p plume-bench --bin gen_fixtures

# 3. Run the harness tests.
cargo test -p plume-bench
```

## Limitations (and they will grow before they shrink)

- No SQL frontend, ever. Query plans are constructed in Rust.
- No memory budget on the hash aggregator — wide-key GROUP BY workloads will
  OOM. Spilling is on the Week-13 stretch list.
- No joins yet (Week 13 stretch, scope-limited if attempted).
- Single-threaded execution.

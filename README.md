# plume

A small, programmatic-API query engine in Rust. No SQL parser — the plan **is**
the API. Built to focus on execution quality (Arrow kernels, Parquet-aware
scans, a real optimizer) rather than SQL surface area.

Every query is diff-tested against DuckDB on the same input, so the
correctness story is mechanical: if `cargo test` is green, plume's results
agree with DuckDB to ~10 significant digits on every supported pipeline.

```rust
use plume_core::{avg, col, count_star, lit, sum};
use plume_exec::collect;
use plume_io::scan_parquet_auto;
use plume_optimizer::Optimizer;

let plan = scan_parquet_auto("fixtures/sales.parquet")?
    .filter(col("price").gt(lit(500.0)))?
    .aggregate(
        vec![col("region")],
        vec![
            count_star().alias("n"),
            avg(col("price")).alias("avg_price"),
            sum(col("quantity")).alias("total_qty"),
        ],
    )?
    .into_plan();

let optimized = Optimizer::default().optimize(plan)?;
let batches   = collect(&optimized)?;
```

The optimizer pushes `price > 500` into the Parquet reader's row-group
filter (skipping ~half the file) and prunes the scan to just the three
columns that flow through (`region`, `price`, `quantity`).

## What's in the box

**Plan layer** (`plume-core`) — `LogicalPlan` enum (`Scan`, `Filter`,
`Projection`, `Aggregate`, `Limit`), `Expr` enum with strict
schema-time type checking, fluent builder API. Every node knows its
output schema; aggregates reject non-aggregate expressions in agg
position at plan-build time.

**Execution** (`plume-exec`) — Streaming `RecordBatch` iterators per
operator. CSV and Parquet sources, Filter / Projection / Limit /
AggregateExec / HashAggregateExec. Compiled-expression hot path
resolves column indices once at construction. Opt-in
`ParallelParquetScanExec` reads row groups in parallel via rayon.

**Optimizer** (`plume-optimizer`) — `OptimizerRule` trait + fixed-point
driver. Default pipeline:

1. **ConstantFolding** — `lit OP lit -> lit`, `true AND x -> x`, etc.
2. **DeadProjectionElim** — drops identity projections.
3. **PredicatePushdown** — pushes `col OP lit` predicates into Parquet
   scans for row-group min/max pruning.
4. **ColumnPruning** — restricts Parquet scans to the columns that
   flow up through the plan.
5. **LimitPushdown** — folds a top-level Limit into the Scan's row cap.

The Filter operator stays in the plan after pushdown — pushdown is a
row-group-skipping hint, not a replacement for filtering.

**I/O** (`plume-io`) — `scan_parquet_auto(path)` reads schema from
metadata so callers don't spell out 5 `Field` declarations.

**Test harness** (`plume-bench`) — `assert_matches_duckdb(plan_output,
sql, parquet_files, csv_files, order_sensitive)` writes fixtures,
drives both engines, and renders results to a canonical row-of-strings
form before comparing. Float comparisons use scientific notation with
~10 significant digits to absorb summation-order drift on million-row
SUM/AVG.

## Locked scope decisions (from the start)

- **Lazy evaluation only.** Plans are built, then `.collect()` executes.
- **Arrow as the only type system.** No parallel type hierarchy.
- **Owned data, not zero-copy.** Optimize allocations later, with profiler evidence.
- **Single-threaded by default.** Parallel scan is opt-in via `ParallelParquetScanExec`.
- **Correctness over performance.** Every query is diffed against DuckDB.

## Layout

```
crates/
  plume-core/        LogicalPlan, Expr, DataFrame builder
  plume-exec/        ExecutionPlan, operators, scalar evaluator
  plume-optimizer/   OptimizerRule trait + 5 rules + fixed-point driver
  plume-io/          Parquet schema reader, scan_parquet_auto
  plume-bench/       DuckDB diff harness, fixture gen, criterion benches
fixtures/            (gitignored) generated test data
examples/            wired through plume-bench (see below)
docs/                perf.md (bench methodology + cost ranking)
```

## Quickstart

```sh
# Cold build (~3–5 min — bundled DuckDB compiles from source).
cargo build --workspace

# Generate fixtures: 10K-row people.csv + 1M-row sales.parquet
# with 8 disjoint price-band row groups (so pushdown can demonstrably prune).
cargo run -p plume-bench --bin gen_fixtures

# Run the full test suite (168 tests as of Week 13).
cargo test --workspace

# Try the demos.
cargo run -p plume-bench --example csv_aggregate
cargo run -p plume-bench --example parquet_filter
cargo run -p plume-bench --example group_by_pipeline
cargo run -p plume-bench --example optimizer_demo
```

For benches: `cargo bench -p plume-bench`. See [docs/perf.md](docs/perf.md)
for what each bench group measures and how to read the output.

## Limitations

- **No SQL frontend, ever.** Plans are constructed in Rust. This is
  a deliberate scope decision — the engine is the part that's
  interesting; SQL parsers are well-trod ground.
- **No joins.** Out of scope for this build.
- **No memory budget on the hash aggregator.** Pathological-cardinality
  GROUP BY workloads will OOM. Spilling would be a separate project.
- **Row-at-a-time hash GROUP BY.** Acknowledged in
  [docs/perf.md](docs/perf.md) as the next obvious optimization;
  vectorized group-by (hash whole columns at once) is the better path
  but isn't in this branch.
- **No predicate pushdown for CSV.** CSV is row-oriented, so predicate
  pushdown only saves parse CPU, not I/O. The optimizer skips CSV
  scans on purpose.
- **NotEq pushdown is conservative.** Row-group stats can rarely prove
  `col != lit` is false for every row in the group, so we keep the
  group. Eq / Lt / LtEq / Gt / GtEq prune as you'd expect.
- **Float NaN in row-group stats** disables pruning for that group:
  arrow's stats are `Option<f64>` and a NaN min/max would corrupt the
  bound check.

## Resume-relevant numbers

The 1M-row `sales.parquet` fixture is laid out as 8 row groups of
125,000 rows each, with prices in disjoint bands of 125 each. The
pushdown test asserts that `price > 950` reads exactly 125,000 rows
through the scan (only the [875, 1000) band survives) — proof that
min/max stats actually skip work, not just look up correct.

```
$ cargo test -p plume-bench --test pushdown_diff pushdown_demonstrably_skips_row_groups -- --nocapture
test pushdown_demonstrably_skips_row_groups ... ok
```

## Verification

Every supported pipeline is checked against DuckDB on the same input:

| Surface             | Tests                                                                                |
|---------------------|--------------------------------------------------------------------------------------|
| Scalar evaluator    | 24 DuckDB-diff tests covering comparisons, arithmetic, three-valued AND/OR           |
| CSV pipelines       | 8 end-to-end diffs on people.csv (filter, project, limit, arithmetic projection)     |
| Global aggregates   | 8 diffs covering COUNT/SUM/AVG/MIN/MAX with promotion (Int32 → Int64, etc.)           |
| GROUP BY            | 7 diffs on people.csv: one-/two-key, post-filter, NULL bucketing, MIN/MAX strings    |
| Parquet pipelines   | 7 diffs on sales.parquet mirroring the CSV surface via `scan_parquet_auto`           |
| Column pruning      | 9 plan-structure tests + 5 three-way diffs (unpruned vs pruned vs DuckDB)            |
| Predicate pushdown  | 9 plan-structure tests + 6 three-way diffs + a row-count assertion proving pruning   |
| Optimizer driver    | 14 framework tests (rule names, fixed-point convergence, fold/elim/limit-pushdown)   |
| Compiled exprs      | 7 unit tests (column index resolution, alias stripping, error paths, null handling) |
| Parallel scan       | 5 tests vs sequential, including pushdown-honouring and projection                   |
| Operators (unit)    | 7 + 10 + 7 + 17 unit tests on the in-memory MemoryExec source                        |

**Total: 168 tests.** No warnings.

## Where the work lives

| Week | Commit         | Subject                                                                       |
|------|----------------|-------------------------------------------------------------------------------|
| 1    | `cd3eeda`      | Workspace + DuckDB diff harness + fixture generator                           |
| 2    | `57cc4f0`      | LogicalPlan + Expr + DataFrame builder + schema propagation                   |
| 3    | `653b4ca`      | Scalar evaluator (Arrow kernels, Kleene three-valued logic)                   |
| 4    | `8241ac5`      | CsvScanExec + FilterExec + ProjectionExec + `.collect()`                      |
| 5    | `4af4497`      | Global aggregation: COUNT, SUM, AVG, MIN, MAX with type promotion             |
| 6    | `c0b44b9`      | HashAggregateExec — GROUP BY                                                  |
| 7    | `78f49df`      | ParquetScanExec + `scan_parquet_auto`                                         |
| 8    | `83ca407`      | Column pruning optimizer pass + ProjectionMask wiring                         |
| 9    | `fa50c32`      | Predicate pushdown + Parquet row-group min/max pruning                        |
| 10   | `9fc8d77`      | OptimizerRule trait + 5-rule pipeline (constant folding, dead-proj, limit)    |
| 11   | `61b2295`      | Compiled expressions (resolved column indices, no per-batch tree walk)        |
| 12   | `e28b7f8`      | Bench methodology + cost ranking ([docs/perf.md](docs/perf.md))               |
| 13   | `80323cb`      | ParallelParquetScanExec — rayon-driven row-group parallelism                  |

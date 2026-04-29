# Performance — methodology and what to look at

This document records how plume's benchmarks are set up, what they measure,
and what to look for when reading the numbers. Concrete results are
machine-dependent, so they live in `target/criterion/` after a run rather
than being checked in.

## How to run

```sh
# 1. Generate fixtures (10K-row CSV, 1M-row Parquet with 8 disjoint price-band row groups)
cargo run -p plume-bench --bin gen_fixtures

# 2. Run the criterion suite
cargo bench -p plume-bench

# Open the HTML report
open target/criterion/report/index.html
```

The first `cargo bench` is slow — it rebuilds DuckDB and the arrow stack
in release mode (~15 GiB of intermediate artifacts on disk). Subsequent
runs are incremental.

## Bench groups

`crates/plume-bench/benches/queries.rs` defines two groups:

### `pipelines/*` — end-to-end queries on `sales.parquet` (1M rows, 8 row groups)

| Bench                  | Plan                                                                     | What it stresses                                                                                                         |
|------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `project_unoptimized`  | scan + project [order_id, price]                                         | Baseline parquet read of all 5 columns + Arrow → projection                                                              |
| `project_optimized`    | optimize() applied to the same plan                                      | Column pruning (W8) drops 3 of 5 columns from the parquet read                                                           |
| `filter_unoptimized`   | scan + filter `price > 950.0` + project                                  | Reads all 1M rows, filters down to ~50K (band width is random above 950)                                                |
| `filter_optimized`     | optimize() applied                                                       | Column pruning + predicate pushdown (W9): row-group min/max stats skip 7 of 8 row groups, leaving ~125K rows decoded     |
| `groupby_unoptimized`  | scan + filter `price > 500.0` + GROUP BY region with COUNT/AVG/SUM       | HashAggregateExec on ~half the file                                                                                       |
| `groupby_optimized`    | optimize() applied                                                       | Pushdown (4 of 8 row groups skipped) + column pruning (only price/region/quantity)                                       |

The unoptimized/optimized pairs are the resume number: they show the
optimizer doing meaningful work on a measurable workload.

### `expression_evaluator/*` — scalar evaluator hot path

| Bench       | What it does                                                                                       |
|-------------|----------------------------------------------------------------------------------------------------|
| `treewalk`  | `evaluate(&Expr, &batch)` — walks the tree, looks up columns by name in a HashMap on every call    |
| `compiled`  | `CompiledExpr::evaluate(&batch)` — column indices pre-resolved at construction (W11)               |

Both run the same predicate `(price > 500.0) AND (region = 'west')` over
a synthetic 8192-row batch. The compiled path drops a HashMap lookup per
column reference per batch. The whole expression has 2 column refs and
2 literals, so per-batch overhead the compile pass eliminates is roughly
`2 * (HashMap::get + Arc<dyn Array> clone)` worth of work — which the
arrow-kernel call itself dwarfs once the batch is non-trivial.

## What's likely to dominate a real query

Reading code, here's the order I'd expect on the 1M-row sales workload,
hottest first:

1. **Parquet decode + decompression** — the parquet crate's batch reader
   is doing dictionary decode, RLE expansion, type conversion. Most of
   the wall clock on a non-pruning query lives here.
2. **Hash GROUP BY probe-and-update** (`HashAggregateExec`). The current
   implementation is row-at-a-time: each row is hashed into a key
   `Vec<KeyValue>`, looked up in a `HashMap`, and the per-aggregate
   accumulators are updated with a 1-row sliced `RecordBatch`. The plan
   acknowledges this in the W11 README note: vectorized group-by (hash
   whole columns at once) is the known faster path.
3. **Filter mask construction** — Arrow's cmp / boolean kernels are
   already SIMD-friendly. The `apply_binary` dispatch in the evaluator
   does one match arm per call; small.
4. **Column tree-walk overhead** — what W11's compiled exprs eliminated.
   On the synthetic 8192-row evaluator bench it's measurable; on a query
   that spends most of its time in parquet decode, it's invisible.

A reasonable next optimization (not in this branch) is the row-at-a-time
hash GROUP BY. The fix is to evaluate group keys for the whole batch up
front, build a `HashMap<key, Vec<row_index>>` once per batch, then call
each accumulator's `update_batch` on the indexed sub-batch instead of
calling `update_batch` once per row. That trades per-row HashMap traffic
for per-group sub-batch construction; for high-cardinality groupings the
crossover hurts, but for the typical low-cardinality case (8 distinct
values for `region`) the speedup is large.

## Reading criterion output

Each bench prints something like:

```
pipelines/filter_optimized
                        time:   [12.4 ms 12.6 ms 12.8 ms]
                        thrpt:  [78.1 Melem/s 79.4 Melem/s 80.6 Melem/s]
```

The throughput line is set to "elements per second" with each element =
one input row, so 80 Melem/s on the 1M-row query means ~12.5 ms wall.

For the filter pair specifically: if `filter_unoptimized` is much slower
than `filter_optimized`, that delta is the W9 row-group pruning paying
off — 1M rows decoded vs ~125K. The difference is what gets cited in the
README.

## What this doesn't do

- No flamegraph capture in the harness. `cargo flamegraph --bench queries`
  works on Linux and `samply record` works on macOS; results are
  machine-specific so the harness doesn't try to bake them in.
- No regression guard. Criterion tracks comparisons across runs locally
  via `target/criterion/`, but there's no CI check that a refactor
  didn't slow `pipelines/*` down by more than X%. That's a future
  add-on if/when this turns into a maintained project.
- No memory/allocation profiling. `dhat-rs` would slot in here if any of
  the row-at-a-time paths show up as the bottleneck.

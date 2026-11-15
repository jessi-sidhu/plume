# plume

A query engine in Rust without a SQL frontend. You build query plans by
chaining methods on a `DataFrame`, and `.collect()` runs them. Every query
is checked against DuckDB on the same input, so if `cargo test` is green
the results agree.

```rust
let plan = scan_parquet_auto("fixtures/sales.parquet")?
    .filter(col("price").gt(lit(500.0)))?
    .aggregate(
        vec![col("region")],
        vec![count_star().alias("n"), avg(col("price")).alias("avg_price")],
    )?
    .into_plan();

let batches = collect(&Optimizer::default().optimize(plan)?)?;
```

About 4,800 lines of Rust across five crates, 208 tests, no warnings.

## What pruning looks like

The 1M-row `sales.parquet` fixture is laid out as 8 row groups of 125,000
rows each, with prices in disjoint bands. When the optimizer pushes
`price > 950` down, only one row group's max value can possibly satisfy
it, so the parquet reader skips the other seven entirely.

This is a best case by construction: `gen_fixtures` builds the file so
the bands don't overlap. For what pruning does on real data, see
[Results](#results).

```
$ cargo run -p plume-bench --example parquet_filter
=== optimized plan (note pushdown=[(price > 950)] on the scan) ===
Limit: 20
  Filter: (price > 950)
    ParquetScan: fixtures/sales.parquet, projection=[order_id, price], pushdown=[(price > 950)]

scan rows without pushdown: 1000000
scan rows with    pushdown: 125000
(pushdown skipped 7 of 8 row groups; the filter still does the per-row check)
```

The Filter stays in the plan after pushdown because row-group min/max
stats only prove that *no* match exists in a group. They can't prove
that *every* row in the surviving group matches, so per-row filtering
still runs.

## Results

How much of a Parquet file does min/max row-group pruning skip on real
data, how much does data layout change that, and does TPC-H understate
it the way the [Snowflake pruning paper](https://arxiv.org/abs/2504.11540)
reports (§8.3)?

[`docs/results-phase1.md`](docs/results-phase1.md) answers that with
TPC-H SF1 and six months of NYC taxi trips, each written shuffled,
sorted, and sorted on two columns. Every number there is generated from
`results/phase1/` by `scripts/report.py`.

## How nulls work

SQL's three-valued logic is the kind of thing you get wrong if you write
your own kernel loops. plume delegates to Arrow's `boolean::and_kleene`
and `boolean::or_kleene` for `AND` and `OR`, then diffs the result
against DuckDB on every truth-table combination:

| a     | b     | a AND b | a OR b |
|-------|-------|---------|--------|
| true  | true  | true    | true   |
| true  | NULL  | NULL    | true   |
| false | NULL  | false   | NULL   |
| NULL  | NULL  | NULL    | NULL   |

The evaluator's tests run all nine `(true/false/NULL)^2` combinations
through both engines. Comparisons against literals propagate nulls the
same way: `NULL > 5` is `NULL`, not `false`.

## How a query becomes batches

```mermaid
flowchart LR
    A[DataFrame builder] --> B[LogicalPlan]
    B --> C[Optimizer]
    C --> D[ExecutionPlan]
    D --> E[BatchStream]
```

The optimizer is a fixed-point driver over five rules:

```mermaid
flowchart LR
    F[ConstantFolding] --> G[DeadProjectionElim]
    G --> H[PredicatePushdown]
    H --> I[ColumnPruning]
    I --> J[LimitPushdown]
```

Order matters. Predicate pushdown attaches predicate columns to the scan,
which column pruning then has to keep in the projection. Limit pushdown
runs last because it folds into the already-projected scan.

## The optimizer in action

`cargo run -p plume-bench --example optimizer_demo` builds a deliberately
wasteful plan and shows the rules grinding it down:

```
BEFORE:
Limit: 50
  Aggregate: groupBy=[region] aggs=[total_qty, n]
    Filter: (price > (100 + 50))
      Projection: [order_id, category, region, price, quantity]
        ParquetScan: fixtures/sales.parquet [5 cols]

AFTER:
Limit: 50
  Aggregate: groupBy=[region] aggs=[total_qty, n]
    Filter: (price > 150)
      ParquetScan: fixtures/sales.parquet, projection=[region, price, quantity], pushdown=[(price > 150)]
```

Four rules fired. ConstantFolding evaluated `100 + 50` at plan time.
DeadProjectionElim removed the SELECT-* projection. PredicatePushdown
attached `(price > 150)` to the scan. ColumnPruning dropped `order_id`
and `category` from the read. LimitPushdown sat out because an Aggregate
sits between the Limit and the Scan, and aggregates change row count, so
folding the limit past it would silently change the query's meaning.

There's a fifth case worth seeing: with a plan shaped
`Limit -> Projection -> Scan`, ColumnPruning narrows the scan,
DeadProjectionElim then drops the now-redundant Projection, and finally
LimitPushdown folds the limit into the scan. Three rules cooperate to
reach a state none of them produce in isolation. That's why the driver
runs in a fixed point rather than a single sequential pass.

## A grouped query end to end

```
$ cargo run -p plume-bench --example group_by_pipeline
+--------+-------------+-------+-----------+-------------------+--------------------+-------------------+
| region | category    | n     | total_qty | avg_price         | min_price          | max_price         |
+--------+-------------+-------+-----------+-------------------+--------------------+-------------------+
| west   | garden      | 24971 |    248940 | 749.6730057611747 | 500.0315956713758  | 999.9811564049711 |
| north  | toys        | 24877 |    248949 | 750.7933018230428 | 500.02438141010185 | 999.9966163968268 |
| east   | electronics | 25366 |    254022 | 750.9629128771891 | 500.0075740129966  | 999.9930958184789 |
| ...    | ...         | ...   | ...       | ...               | ...                | ...               |
+--------+-------------+-------+-----------+-------------------+--------------------+-------------------+
```

The plan is `scan -> filter (price > 500) -> aggregate by (region, category)
with COUNT/SUM/AVG/MIN/MAX`. The optimizer turns the scan into
`projection=[category, region, price, quantity], pushdown=[(price > 500)]`,
so half the row groups are skipped and only four columns are decoded.
Twenty distinct (region, category) groups come out, each with five
aggregates per row. DuckDB returns the same numbers when given the same
parquet, modulo summation-order drift past the 10th significant digit.

## The crates

`plume-core` (~990 lines). `LogicalPlan` and `Expr` enums, the
`DataFrame` builder, schema-time type checking. Every plan node knows
its output schema. Aggregate nodes reject non-aggregate expressions in
agg position at plan-build time, before anything runs. The `Join`
variant is the one multi-input node. Schemas concatenate left then
right with collision suffixes (`id`, `id_1`), and nullability widens
on whichever side an outer join may emit unmatched rows for.

`plume-exec` (~2,500 lines). The execution side. `ExecutionPlan` trait
returning a `Box<dyn Iterator<Item = Result<RecordBatch>> + Send>`,
operators for CSV/Parquet scans, filter, projection, limit, global and
hash-grouped aggregates, hash join in all six equi shapes
(Inner/Left/Right/Full/Semi/Anti), and cross join. The compiled-
expression hot path lives here: column names get resolved to indices
once at operator construction so the per-batch loop is a vec lookup
instead of a HashMap probe. Opt-in `ParallelParquetScanExec` reads
row groups in parallel via rayon.

`plume-optimizer` (~840 lines). Five rules plus a fixed-point driver
that compares plans by `PartialEq` and caps at 8 passes. Each rule
implements `OptimizerRule` (`name()` + `rewrite(plan) -> Result<plan>`).
Two of the rules are join-aware: PredicatePushdown splits AND-conjuncts
above a join into per-side buckets and refuses to push a right-only
predicate through a Left join (it would discard NULL-padded rows).
ColumnPruning splits the parent's required column set across the two
inputs based on output position.

`plume-io` (30 lines). One file. Reads parquet metadata to produce a
`SchemaRef`, then constructs a `DataFrame` rooted at a Parquet scan.
This is the bridge that lets the diff harness write
`scan_parquet_auto(path)` instead of spelling out five `Field`s.

`plume-bench`. Diff harness, fixture generator, criterion benches,
runnable demos. The harness is the load-bearing piece for correctness
(see below).

## How correctness is checked

The harness signature is `assert_matches_duckdb(plume_output, sql,
csv_files, parquet_files, order_sensitive)`. It registers the named
files as DuckDB views, runs the SQL, pulls the result back as Arrow
batches, and renders both sides to a canonical row-of-strings form
before comparing. The string form sidesteps three classes of fake
mismatches:

- Float values with different bit patterns that print identically.
  Floats render in scientific notation with about 10 significant
  digits, which absorbs summation-order drift on million-row SUM/AVG
  without making genuinely different values look the same.
- Dictionary-encoded vs plain string arrays. Both render to the
  same string.
- Schema metadata (field nullability flags, PARQUET annotations).
  Ignored by the renderer.

`order_sensitive=false` sorts both sides before comparing, used for
GROUP BY queries where hash-table iteration order is undefined.

Of the 208 tests, 79 go through this harness end-to-end against
DuckDB. The rest are unit tests (operator behavior, plan-tree
structure, optimizer rewrites, accumulator math, plan-shape rule
checks) where DuckDB agreement isn't the point.

```
plume-bench    diff harness tests vs DuckDB        79
               (filter, project, aggregate, group, prune,
                pushdown, joins inner/outer/semi/anti/cross)
plume-bench    arrow fluency / harness sanity      3
plume-core     DataFrame builder + schema math     24
plume-exec     operators, evaluator, aggregates    56
plume-optimizer rules + driver + join pushdown     46
                                                  ---
                                                   208
```

## Parallel scan

```rust
use plume_exec::{ExecutionPlan, ParallelParquetScanExec};

let scan = ParallelParquetScanExec::new(path, schema)
    .with_pushdown_predicates(vec![col("price").gt(lit(500.0))])
    .with_max_workers(4);
let stream = scan.execute()?;
```

Each surviving row group becomes a rayon task. Workers stream batches
back through an `mpsc` channel, so the foreground iterator pulls them
out in completion order, not file order. Filter / aggregate /
projection don't care about input order. Limit does. The planner
never picks this scan automatically for that reason; you opt in if
you know your downstream operators are order-insensitive.

The pushdown predicate is honoured per-worker: each task reads only
its assigned row groups via `with_row_groups(vec![idx])`, so if
upstream pruning already eliminated half the row groups, the
parallel scan only spawns tasks for the other half.

## Joins

Six equi shapes (Inner / Left / Right / Full / Semi / Anti) plus cross,
all riding on one `HashJoinExec`. Build the right side into a hash
table, probe the left, gather output indices, materialize one batch
per probe-side input via `arrow::compute::take`.

```mermaid
flowchart LR
    R[Right input] -->|drain| B["concat_batches'd build table"]
    B --> H[HashMap of right keys]
    L[Left input] -->|stream| P[probe loop]
    H --> P
    P -->|left_idx, right_idx| T[arrow::compute::take]
    T --> O[output batch]
```

NULL keys never go in the hash table. SQL equi-join treats NULL as
non-matching, so the probe side short-circuits the same way. Outer
joins keep a per-build-row matched bitmap and emit unmatched rows
NULL-padded via `new_null_array` after the probe finishes.

Build a join with the DataFrame:

```rust
let plan = scan_parquet_auto("fixtures/customers.parquet")?
    .join(
        scan_parquet_auto("fixtures/orders.parquet")?,
        vec![(col("id"), col("customer_id"))],
        JoinType::Inner,
    )?
    .filter(col("region").eq(lit("west")))?
    .project(vec![col("name"), col("amount")])?
    .into_plan();
```

`cargo run -p plume-bench --example join_demo` runs that, plus a Left
join with a right-side filter. The optimizer rewrites the inner-join
plan three ways at once:

```
BEFORE:                                    AFTER:
Projection: [name, amount]                 Projection: [name, amount]
  Filter: (region = "west")                  InnerJoin: on=[id = customer_id]
    InnerJoin: on=[id = customer_id]           L:
      L: ParquetScan: customers [3 cols]         Filter: (region = "west")
      R: ParquetScan: orders [4 cols]              ParquetScan: customers,
                                                     pushdown=[(region = "west")]
                                               R:
                                                 ParquetScan: orders,
                                                   projection=[customer_id, amount]
```

Predicate pushdown splits the filter to the customers side. Scan
pushdown then attaches it as a row-group hint. Column pruning narrows
each side independently: customers keeps `id, name, region` (filter +
join key + output), orders keeps `customer_id, amount` (join key +
output). Result table:

```
+-------+--------------------+
| name  | amount             |
+-------+--------------------+
| carol | 734.2330818960052  |
| alice | 817.5468823098956  |
| dan   | 540.9708585589628  |
| bob   | 832.2037180892002  |
| heidi | 611.1012607203518  |
| ...   | ...                |
+-------+--------------------+
```

The second demo is the case where the optimizer has to know when
*not* to fire. A `LEFT JOIN` with `WHERE amount > 500` above it.
Above the join, `amount` is NULL on unmatched rows and `NULL > 500`
is `NULL`, so the filter correctly drops them. Push it through the
join and the right input shrinks, the LEFT join emits a different
set of unmatched rows, and the answer diverges. So it stays put:

```
BEFORE:                                    AFTER:
Filter: (amount > 500)                     Filter: (amount > 500)
  LeftJoin: on=[id = customer_id]            LeftJoin: on=[id = customer_id]
    L: ParquetScan: customers                  L: ParquetScan: customers
    R: ParquetScan: orders                     R: ParquetScan: orders
```

The per-join-type pushdown decisions live as a table in the
doc-comment on `push_filter_through_join` in
[`crates/plume-optimizer/src/lib.rs`](crates/plume-optimizer/src/lib.rs).
Two tests enforce them: `left_join_then_filter_on_right_amount_strips_nulls`
in `join_diff.rs` checks the end-to-end answer matches DuckDB, and
`left_join_does_not_push_right_only_predicate` in `join_pushdown.rs`
checks the rule made the right call at the plan level.

## What's missing

No SQL parser. Plans are Rust. The engine is the part worth building
and SQL parsers are well-trod ground.

No qualified columns or self-joins. Joins disambiguate name collisions
with a numeric suffix (`id`, `id_1`). Referring to a column as
`customers.id` isn't supported. Self-joins would need that to be
useful.

No sort-merge join. Hash join handles every shape supported here.
Sort-merge would matter for build sides too big to fit in memory,
which would also need spill-to-disk, which is also out of scope.

No cost-based join reordering. Joins execute in the order you write
them. The optimizer reorders predicates across joins but not the
joins themselves. Doing that needs cardinality statistics.

The hash GROUP BY is row-at-a-time. Each row gets hashed into a key
`Vec<KeyValue>`, looked up in a `HashMap`, and the per-aggregate
accumulators are updated with a 1-row sliced `RecordBatch`. The
faster path is to evaluate group keys for the whole batch up front,
build a `HashMap<key, Vec<row_index>>` once per batch, then call each
accumulator's `update_batch` on the indexed sub-batch. For
low-cardinality groupings (the common case) the speedup is large. The
writeup is in [`docs/perf.md`](docs/perf.md).

`NotEq` predicates don't prune row groups even in principle. Stats
can prove `col != lit` is always false only if the entire row group
equals `lit`, which they don't track cheaply. The optimizer collects
the predicate but the row-group filter ignores `NotEq`.

No DECIMAL type. The Phase 1 datasets (TPC-H, NYC taxi) have DECIMAL
money columns, so `prepare_datasets` casts every DECIMAL column to
DOUBLE on export and records which columns it cast in
`data/manifest.json`. Sums over those columns can differ from exact
decimal arithmetic in the last few bits.

Float `NaN` in row-group min/max disables pruning for that group.
Comparing against `NaN` returns `false` for every operator, so a
`NaN` bound would silently skip groups that should have been kept.

## Run it

```sh
cargo build --workspace                            # ~3-5 min cold (bundled DuckDB)
cargo run -p plume-bench --bin gen_fixtures        # CSV + parquet fixtures (incl. customers/orders for joins)
cargo test --workspace                             # 208 tests
cargo run -p plume-bench --example join_demo       # or optimizer_demo / csv_aggregate / parquet_filter / group_by_pipeline
cargo run --release -p plume-bench --bin prepare_datasets   # TPC-H SF1 + NYC taxi Jan-Jun 2023 into data/ (gitignored)
cargo run --release -p plume-bench --bin prune_bench        # writes results/phase1/
python3 scripts/report.py                                   # writes docs/results-phase1.md (needs pandas, matplotlib)
```

For benchmark methodology and how to interpret `cargo bench` output,
see [`docs/perf.md`](docs/perf.md).

## Where the interesting code lives

Things worth reading in order:

- `crates/plume-core/src/expr.rs`: the `Expr` enum, type resolution,
  and the strict same-type comparison rule that rejects bad plans
  before they execute.
- `crates/plume-core/src/plan.rs`: `LogicalPlan` including the `Join`
  variant. Schema propagation per node, plus the collision-suffix and
  nullability-widening rules that make join output schemas trustworthy.
- `crates/plume-exec/src/aggregate.rs`: the `Accumulator` trait, six
  concrete impls, and the `HashAggregateExec` row-at-a-time group-by
  loop. Key handling lives in `keys.rs` next door, shared with hash
  join.
- `crates/plume-exec/src/join.rs`: `HashJoinExec` for all six equi
  shapes plus `CrossJoinExec`. The per-row decision matrix and the
  three emission paths (matched, unmatched-left, left-only for
  Semi/Anti) sit at the top of `execute()`.
- `crates/plume-optimizer/src/lib.rs`: five rules and the fixed-point
  driver. The `push_filter_through_join` doc-comment carries the full
  per-join-type pushdown safety table.
- `crates/plume-exec/src/operators.rs`: `row_group_can_satisfy` with
  per-operator pruning rules. The `Eq` case is one line, the rest
  similar.
- `crates/plume-bench/src/harness.rs`: the diff harness. About 165
  lines, including the canonical-row renderer.

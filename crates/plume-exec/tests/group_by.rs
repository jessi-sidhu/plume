//! Unit tests for HashAggregateExec — GROUP BY semantics on small fixtures.
//! End-to-end DuckDB diffs live in plume-bench/tests/group_diff.rs.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{avg, col, count_star, max, min, sum, DataFrame};
use plume_exec::{collect, ExecutionPlan, MemoryExec};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("region", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("amount", DataType::Int32, true),
        Field::new("price", DataType::Float64, true),
    ]))
}

fn batch(
    regions: Vec<&str>,
    categories: Vec<&str>,
    amounts: Vec<Option<i32>>,
    prices: Vec<Option<f64>>,
) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(regions)),
            Arc::new(StringArray::from(categories)),
            Arc::new(Int32Array::from(amounts)),
            Arc::new(Float64Array::from(prices)),
        ],
    )
    .unwrap()
}

/// Run a group-by plan over in-memory batches and pull the single output batch.
fn run(
    batches: Vec<RecordBatch>,
    group_by: Vec<plume_core::Expr>,
    aggs: Vec<plume_core::Expr>,
) -> RecordBatch {
    let mem: Arc<dyn ExecutionPlan> = Arc::new(MemoryExec::new(batches, schema()));
    // Build the LogicalPlan via DataFrame for schema math, then lift the
    // HashAggregateExec manually so we can substitute MemoryExec as the source.
    let df = DataFrame::scan_csv("ignored", schema())
        .aggregate(group_by, aggs)
        .unwrap();
    let plan = df.into_plan();
    let (group_exprs, agg_exprs, out_schema) = match plan {
        plume_core::LogicalPlan::Aggregate {
            group_exprs,
            agg_exprs,
            schema,
            ..
        } => (group_exprs, agg_exprs, schema),
        _ => unreachable!(),
    };
    let exec: Arc<dyn ExecutionPlan> = Arc::new(plume_exec::HashAggregateExec::new(
        mem,
        group_exprs,
        agg_exprs,
        out_schema,
    ));
    exec.execute().unwrap().next().unwrap().unwrap()
}

/// Index into the output batch by the value of the first (string) group
/// column, returning a map name -> row index. Convenient since hash group
/// output order is unspecified.
fn name_index(batch: &RecordBatch, col_idx: usize) -> HashMap<String, usize> {
    let arr = batch.column(col_idx).as_string::<i32>();
    (0..batch.num_rows())
        .map(|r| (arr.value(r).to_string(), r))
        .collect()
}

#[test]
fn single_string_group_with_count_star() {
    let b = batch(
        vec!["west", "east", "west", "east", "west"],
        vec!["a", "a", "b", "b", "a"],
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
    );
    let out = run(vec![b], vec![col("region")], vec![count_star().alias("n")]);
    assert_eq!(out.num_rows(), 2);
    let idx = name_index(&out, 0);
    assert_eq!(
        out.column(1).as_primitive::<Int64Type>().value(idx["west"]),
        3
    );
    assert_eq!(
        out.column(1).as_primitive::<Int64Type>().value(idx["east"]),
        2
    );
}

#[test]
fn group_by_two_strings_emits_one_row_per_combination() {
    let b = batch(
        vec!["west", "east", "west", "east", "west"],
        vec!["a", "a", "b", "b", "a"],
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
    );
    let out = run(
        vec![b],
        vec![col("region"), col("category")],
        vec![sum(col("amount")).alias("total")],
    );
    // (west,a)=6, (east,a)=2, (west,b)=3, (east,b)=4 -> 4 distinct groups
    assert_eq!(out.num_rows(), 4);
    let r = out.column(0).as_string::<i32>();
    let c = out.column(1).as_string::<i32>();
    let totals = out.column(2).as_primitive::<Int64Type>();
    let mut seen: HashMap<(String, String), i64> = HashMap::new();
    for i in 0..out.num_rows() {
        seen.insert(
            (r.value(i).to_string(), c.value(i).to_string()),
            totals.value(i),
        );
    }
    assert_eq!(seen[&("west".into(), "a".into())], 6);
    assert_eq!(seen[&("east".into(), "a".into())], 2);
    assert_eq!(seen[&("west".into(), "b".into())], 3);
    assert_eq!(seen[&("east".into(), "b".into())], 4);
}

#[test]
fn group_by_aggregates_split_across_batches_and_merge() {
    // Two batches with overlapping groups — accumulator state must persist
    // between batches.
    let b1 = batch(
        vec!["w", "e"],
        vec!["x", "x"],
        vec![Some(10), Some(20)],
        vec![Some(1.0), Some(2.0)],
    );
    let b2 = batch(
        vec!["w", "w"],
        vec!["x", "x"],
        vec![Some(30), Some(40)],
        vec![Some(3.0), Some(4.0)],
    );
    let out = run(
        vec![b1, b2],
        vec![col("region")],
        vec![
            sum(col("amount")).alias("total"),
            avg(col("price")).alias("avg_price"),
        ],
    );
    assert_eq!(out.num_rows(), 2);
    let idx = name_index(&out, 0);
    let totals = out.column(1).as_primitive::<Int64Type>();
    let avgs = out.column(2).as_primitive::<Float64Type>();
    assert_eq!(totals.value(idx["w"]), 80); // 10+30+40
    assert_eq!(totals.value(idx["e"]), 20);
    assert!((avgs.value(idx["w"]) - (1.0 + 3.0 + 4.0) / 3.0).abs() < 1e-12);
    assert!((avgs.value(idx["e"]) - 2.0).abs() < 1e-12);
}

#[test]
fn nulls_in_grouped_aggregates_skip_per_sql_semantics() {
    // amount=NULL rows must not count toward SUM/AVG; COUNT(*) still includes them.
    let b = batch(
        vec!["w", "w", "e", "e"],
        vec!["x", "x", "x", "x"],
        vec![Some(10), None, Some(30), None],
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
    );
    let out = run(
        vec![b],
        vec![col("region")],
        vec![
            count_star().alias("n"),
            sum(col("amount")).alias("total"),
            avg(col("amount")).alias("avg_amt"),
        ],
    );
    let idx = name_index(&out, 0);
    let n = out.column(1).as_primitive::<Int64Type>();
    let total = out.column(2).as_primitive::<Int64Type>();
    let avg_arr = out.column(3).as_primitive::<Float64Type>();
    assert_eq!(n.value(idx["w"]), 2);
    assert_eq!(total.value(idx["w"]), 10); // null skipped
    assert!((avg_arr.value(idx["w"]) - 10.0).abs() < 1e-12);
    assert_eq!(n.value(idx["e"]), 2);
    assert_eq!(total.value(idx["e"]), 30);
    assert!((avg_arr.value(idx["e"]) - 30.0).abs() < 1e-12);
}

#[test]
fn null_group_key_forms_its_own_bucket() {
    // Build a batch where the *category* column is nullable and contains nulls.
    let s: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("amount", DataType::Int32, false),
    ]));
    let b = RecordBatch::try_new(
        s.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                Some("a"),
                None,
                Some("a"),
                None,
                Some("b"),
            ])),
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
        ],
    )
    .unwrap();
    let mem: Arc<dyn ExecutionPlan> = Arc::new(MemoryExec::new(vec![b], s.clone()));
    let df = DataFrame::scan_csv("ignored", s)
        .aggregate(
            vec![col("category")],
            vec![sum(col("amount")).alias("total")],
        )
        .unwrap();
    let (g, a, out_s) = match df.into_plan() {
        plume_core::LogicalPlan::Aggregate {
            group_exprs,
            agg_exprs,
            schema,
            ..
        } => (group_exprs, agg_exprs, schema),
        _ => unreachable!(),
    };
    let exec: Arc<dyn ExecutionPlan> =
        Arc::new(plume_exec::HashAggregateExec::new(mem, g, a, out_s));
    let out = exec.execute().unwrap().next().unwrap().unwrap();

    // Three groups: 'a' (1+3=4), 'b' (5), NULL (2+4=6)
    assert_eq!(out.num_rows(), 3);
    let cat = out.column(0).as_string::<i32>();
    let total = out.column(1).as_primitive::<Int64Type>();
    let mut by_key: HashMap<Option<String>, i64> = HashMap::new();
    for i in 0..out.num_rows() {
        let k = if cat.is_null(i) {
            None
        } else {
            Some(cat.value(i).to_string())
        };
        by_key.insert(k, total.value(i));
    }
    assert_eq!(by_key[&Some("a".into())], 4);
    assert_eq!(by_key[&Some("b".into())], 5);
    assert_eq!(by_key[&None], 6);
}

#[test]
fn group_by_int_column_works() {
    let s: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("bucket", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]));
    let b = RecordBatch::try_new(
        s.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 2, 1, 3])),
            Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60])),
        ],
    )
    .unwrap();
    let mem: Arc<dyn ExecutionPlan> = Arc::new(MemoryExec::new(vec![b], s.clone()));
    let df = DataFrame::scan_csv("ignored", s)
        .aggregate(
            vec![col("bucket")],
            vec![min(col("v")).alias("mn"), max(col("v")).alias("mx")],
        )
        .unwrap();
    let (g, a, out_s) = match df.into_plan() {
        plume_core::LogicalPlan::Aggregate {
            group_exprs,
            agg_exprs,
            schema,
            ..
        } => (group_exprs, agg_exprs, schema),
        _ => unreachable!(),
    };
    let exec: Arc<dyn ExecutionPlan> =
        Arc::new(plume_exec::HashAggregateExec::new(mem, g, a, out_s));
    let out = exec.execute().unwrap().next().unwrap().unwrap();
    assert_eq!(out.num_rows(), 3);
    let buckets = out.column(0).as_primitive::<Int32Type>();
    let mns = out.column(1).as_primitive::<Int32Type>();
    let mxs = out.column(2).as_primitive::<Int32Type>();
    let mut by_b: HashMap<i32, (i32, i32)> = HashMap::new();
    for i in 0..out.num_rows() {
        by_b.insert(buckets.value(i), (mns.value(i), mxs.value(i)));
    }
    assert_eq!(by_b[&1], (10, 50));
    assert_eq!(by_b[&2], (20, 40));
    assert_eq!(by_b[&3], (60, 60));
}

#[test]
fn collect_runs_full_group_by_pipeline_through_planner() {
    // Whole pipeline end-to-end: CSV scan -> filter -> group by -> aggregate.
    use std::io::Write;
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("data.csv");
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "g,v").unwrap();
    for (g, v) in [("a", 1), ("b", 2), ("a", 3), ("b", 4), ("a", 5), ("c", 99)] {
        writeln!(f, "{g},{v}").unwrap();
    }
    drop(f);
    let s: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Utf8, false),
        Field::new("v", DataType::Int32, false),
    ]));
    let plan = DataFrame::scan_csv(&path, s)
        .filter(col("v").lt(plume_core::lit(10_i32)))
        .unwrap()
        .aggregate(vec![col("g")], vec![sum(col("v")).alias("total")])
        .unwrap()
        .into_plan();
    let out = collect(&plan).unwrap();
    let b = &out[0];
    // After filter: a=1+3+5=9, b=2+4=6. c=99 dropped.
    assert_eq!(b.num_rows(), 2);
    let g = b.column(0).as_string::<i32>();
    let total = b.column(1).as_primitive::<Int64Type>();
    let mut by_g: HashMap<String, i64> = HashMap::new();
    for i in 0..b.num_rows() {
        by_g.insert(g.value(i).to_string(), total.value(i));
    }
    assert_eq!(by_g["a"], 9);
    assert_eq!(by_g["b"], 6);
}

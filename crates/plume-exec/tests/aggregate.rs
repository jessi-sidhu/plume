//! Direct (non-harness) tests for global aggregation accumulators.
//! End-to-end DuckDB diffs live in plume-bench/tests/agg_diff.rs.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{avg, col, count, count_star, max, min, sum, DataFrame};
use plume_exec::{collect, ExecutionPlan, MemoryExec};

fn nums_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Float64, true),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn nums_batch() -> RecordBatch {
    RecordBatch::try_new(
        nums_schema(),
        vec![
            // a: Int32 nullable
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(2),
                None,
                Some(4),
                Some(5),
            ])),
            // b: Float64 nullable
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                Some(2.5),
                Some(3.5),
                None,
                Some(8.0),
            ])),
            // name: Utf8 nullable
            Arc::new(StringArray::from(vec![
                Some("alice"),
                Some("bob"),
                None,
                Some("carol"),
                Some("dan"),
            ])),
        ],
    )
    .unwrap()
}

/// Run an aggregate plan over a single in-memory batch.
fn run_agg(batch: RecordBatch, schema: SchemaRef, aggs: Vec<plume_core::Expr>) -> RecordBatch {
    let mem: Arc<dyn ExecutionPlan> = Arc::new(MemoryExec::new(vec![batch], schema.clone()));

    // Build the LogicalPlan via DataFrame so the schema math matches the planner.
    // We need a scan node that returns the same schema as our MemoryExec; use
    // a CSV scan plan node (path is unused since we're substituting the input
    // execution plan manually below).
    let df = DataFrame::scan_csv("ignored", schema)
        .aggregate(vec![], aggs)
        .unwrap();
    let plan = df.into_plan();

    // Borrow planner internals: use the mem source directly. Simpler than
    // monkey-patching the planner — we lift just the AggregateExec.
    use plume_core::LogicalPlan;
    let (agg_exprs, out_schema) = match plan {
        LogicalPlan::Aggregate {
            agg_exprs, schema, ..
        } => (agg_exprs, schema),
        _ => unreachable!(),
    };
    let agg = plume_exec::AggregateExec::new(mem, agg_exprs, out_schema);
    let agg: Arc<dyn ExecutionPlan> = Arc::new(agg);
    agg.execute().unwrap().next().unwrap().unwrap()
}

#[test]
fn count_star_counts_every_row() {
    let out = run_agg(nums_batch(), nums_schema(), vec![count_star()]);
    assert_eq!(out.num_rows(), 1);
    assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 5);
}

#[test]
fn count_expr_skips_nulls() {
    // a has one null in 5 rows -> COUNT(a) = 4
    let out = run_agg(nums_batch(), nums_schema(), vec![count(col("a"))]);
    assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 4);
}

#[test]
fn sum_int32_promotes_to_int64() {
    // a = [1, 2, NULL, 4, 5] -> SUM = 12 as Int64
    let out = run_agg(nums_batch(), nums_schema(), vec![sum(col("a"))]);
    assert_eq!(out.schema().field(0).data_type(), &DataType::Int64);
    assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 12);
}

#[test]
fn sum_float_returns_float() {
    // b = [1.0, 2.5, 3.5, NULL, 8.0] -> SUM = 15.0 as Float64
    let out = run_agg(nums_batch(), nums_schema(), vec![sum(col("b"))]);
    assert_eq!(out.schema().field(0).data_type(), &DataType::Float64);
    let v = out.column(0).as_primitive::<Float64Type>().value(0);
    assert!((v - 15.0).abs() < 1e-12, "got {v}");
}

#[test]
fn avg_returns_float64_dividing_sum_by_non_null_count() {
    // a non-null = [1, 2, 4, 5], sum 12, count 4, avg 3.0
    let out = run_agg(nums_batch(), nums_schema(), vec![avg(col("a"))]);
    assert_eq!(out.schema().field(0).data_type(), &DataType::Float64);
    let v = out.column(0).as_primitive::<Float64Type>().value(0);
    assert!((v - 3.0).abs() < 1e-12, "got {v}");
}

#[test]
fn min_max_preserve_input_type() {
    let batch = nums_batch();
    let s = nums_schema();
    let out_min = run_agg(batch.clone(), s.clone(), vec![min(col("a"))]);
    assert_eq!(out_min.schema().field(0).data_type(), &DataType::Int32);
    assert_eq!(out_min.column(0).as_primitive::<Int32Type>().value(0), 1);

    let out_max = run_agg(batch, s, vec![max(col("a"))]);
    assert_eq!(out_max.column(0).as_primitive::<Int32Type>().value(0), 5);
}

#[test]
fn min_max_on_strings() {
    let batch = nums_batch();
    let s = nums_schema();
    let out_min = run_agg(batch.clone(), s.clone(), vec![min(col("name"))]);
    assert_eq!(out_min.column(0).as_string::<i32>().value(0), "alice");
    let out_max = run_agg(batch, s, vec![max(col("name"))]);
    assert_eq!(out_max.column(0).as_string::<i32>().value(0), "dan");
}

#[test]
fn aggregate_over_empty_input_returns_null_for_sum_and_zero_for_count() {
    let empty = RecordBatch::new_empty(nums_schema());
    let s = nums_schema();
    let out = run_agg(
        empty,
        s,
        vec![
            count_star(),
            count(col("a")),
            sum(col("a")),
            avg(col("a")),
            min(col("a")),
            max(col("a")),
        ],
    );
    // count_star, count(a) -> 0
    assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 0);
    assert_eq!(out.column(1).as_primitive::<Int64Type>().value(0), 0);
    // sum, avg, min, max -> NULL
    assert!(out.column(2).is_null(0));
    assert!(out.column(3).is_null(0));
    assert!(out.column(4).is_null(0));
    assert!(out.column(5).is_null(0));
}

#[test]
fn aggregate_over_all_null_column_returns_null_for_sum_avg_min_max() {
    let s = nums_schema();
    let batch = RecordBatch::try_new(
        s.clone(),
        vec![
            Arc::new(Int32Array::from(vec![None, None, None] as Vec<Option<i32>>)),
            Arc::new(Float64Array::from(
                vec![None, None, None] as Vec<Option<f64>>
            )),
            Arc::new(StringArray::from(
                vec![None, None, None] as Vec<Option<&str>>
            )),
        ],
    )
    .unwrap();
    let out = run_agg(
        batch,
        s,
        vec![
            count(col("a")),
            sum(col("a")),
            avg(col("a")),
            min(col("a")),
            max(col("a")),
        ],
    );
    assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 0);
    assert!(out.column(1).is_null(0));
    assert!(out.column(2).is_null(0));
    assert!(out.column(3).is_null(0));
    assert!(out.column(4).is_null(0));
}

#[test]
fn full_pipeline_collect_runs_filter_then_aggregate() {
    // Hand-build a CSV file, scan, filter, aggregate via collect().
    use std::io::Write;
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("data.csv");
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "x").unwrap();
    for v in 1..=10 {
        writeln!(f, "{v}").unwrap();
    }
    drop(f);

    let s: SchemaRef = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
    let plan = DataFrame::scan_csv(&path, s)
        .filter(col("x").gt(plume_core::lit(5_i32)))
        .unwrap()
        .aggregate(vec![], vec![sum(col("x")), avg(col("x")), count_star()])
        .unwrap()
        .into_plan();

    let out = collect(&plan).unwrap();
    let b = &out[0];
    // x>5 is [6,7,8,9,10]; sum=40, avg=8.0, count=5
    assert_eq!(b.column(0).as_primitive::<Int64Type>().value(0), 40);
    let avg_val = b.column(1).as_primitive::<Float64Type>().value(0);
    assert!((avg_val - 8.0).abs() < 1e-12, "got {avg_val}");
    assert_eq!(b.column(2).as_primitive::<Int64Type>().value(0), 5);
}

//! Unit tests for execution operators using MemoryExec as the source.
//! End-to-end CSV diffs against DuckDB live in plume-bench/tests/csv_diff.rs.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use plume_core::{col, lit};
use plume_exec::{
    collect, create_physical_plan, ExecutionPlan, FilterExec, LimitExec, MemoryExec,
    ProjectionExec,
};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, true),
    ]))
}

fn batch(ids: &[i32], names: &[&str], ages: Vec<Option<i32>>) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(names.to_vec())),
            Arc::new(Int32Array::from(ages)),
        ],
    )
    .unwrap()
}

fn drain(exec: Arc<dyn ExecutionPlan>) -> Vec<RecordBatch> {
    exec.execute().unwrap().collect::<Result<Vec<_>, _>>().unwrap()
}

#[test]
fn memory_exec_round_trips_input_batches() {
    let b = batch(&[1, 2, 3], &["a", "b", "c"], vec![Some(20), None, Some(40)]);
    let exec: Arc<dyn ExecutionPlan> = Arc::new(MemoryExec::new(vec![b.clone()], schema()));
    let out = drain(exec);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].num_rows(), 3);
}

#[test]
fn filter_exec_drops_non_matching_rows_and_propagates_nulls() {
    let b = batch(&[1, 2, 3, 4], &["a", "b", "c", "d"], vec![Some(20), None, Some(40), Some(50)]);
    let mem: Arc<dyn ExecutionPlan> = Arc::new(MemoryExec::new(vec![b], schema()));
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::new(mem, col("age").gt(lit(25_i32))));
    let out = drain(filter);
    let ids: Vec<i32> = out
        .iter()
        .flat_map(|b| b.column_by_name("id").unwrap().as_primitive::<Int32Type>().iter().flatten())
        .collect();
    // age > 25: id=3 (40), id=4 (50). id=2 (NULL) is dropped (NULL is not true).
    assert_eq!(ids, vec![3, 4]);
}

#[test]
fn projection_exec_reorders_and_renames_columns() {
    let b = batch(&[1, 2], &["a", "b"], vec![Some(20), Some(30)]);
    let mem: Arc<dyn ExecutionPlan> = Arc::new(MemoryExec::new(vec![b], schema()));
    // Build via DataFrame so the schema math is the same path the planner uses.
    let df = plume_core::DataFrame::scan_csv("ignored", schema())
        .project(vec![col("name").alias("who"), col("age").plus(lit(1_i32)).alias("age_next")])
        .unwrap();
    let proj_schema = df.schema();
    let proj: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::new(
        mem,
        vec![col("name").alias("who"), col("age").plus(lit(1_i32)).alias("age_next")],
        proj_schema.clone(),
    ));
    let out = drain(proj);
    assert_eq!(out.len(), 1);
    let b = &out[0];
    assert_eq!(b.schema().field(0).name(), "who");
    assert_eq!(b.schema().field(1).name(), "age_next");
    let ages = b.column(1).as_primitive::<Int32Type>();
    assert_eq!(ages.value(0), 21);
    assert_eq!(ages.value(1), 31);
}

#[test]
fn limit_exec_caps_total_rows_across_batch_boundaries() {
    // Three small batches of 2 rows each (6 total). Limit 3 should cut the
    // last batch in half.
    let b1 = batch(&[1, 2], &["a", "b"], vec![Some(1), Some(2)]);
    let b2 = batch(&[3, 4], &["c", "d"], vec![Some(3), Some(4)]);
    let b3 = batch(&[5, 6], &["e", "f"], vec![Some(5), Some(6)]);
    let mem: Arc<dyn ExecutionPlan> = Arc::new(MemoryExec::new(vec![b1, b2, b3], schema()));
    let lim: Arc<dyn ExecutionPlan> = Arc::new(LimitExec::new(mem, 3));
    let out = drain(lim);
    let total: usize = out.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3);
    // First batch full (2 rows); second trimmed to 1.
    assert_eq!(out[0].num_rows(), 2);
    assert_eq!(out[1].num_rows(), 1);
}

#[test]
fn planner_rejects_aggregate_for_now() {
    let df = plume_core::DataFrame::scan_csv("ignored", schema())
        .aggregate(vec![col("name")], vec![plume_core::sum(col("age"))])
        .unwrap();
    match create_physical_plan(&df.into_plan()) {
        Err(plume_exec::EvalError::NotImplemented(_)) => {}
        other => panic!("expected NotImplemented, got {other:?}", other = other.err()),
    }
}

#[test]
fn planner_rejects_parquet_scan_for_now() {
    let df = plume_core::DataFrame::scan_parquet("nope.parquet", schema());
    match create_physical_plan(&df.into_plan()) {
        Err(plume_exec::EvalError::NotImplemented(_)) => {}
        other => panic!("expected NotImplemented, got {other:?}", other = other.err()),
    }
}

#[test]
fn collect_runs_a_csv_scan_filter_project_pipeline() {
    // Materialize a tiny CSV in tempdir, then run plume end-to-end.
    use std::io::Write;
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("p.csv");
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "id,name,age").unwrap();
    writeln!(f, "1,alice,30").unwrap();
    writeln!(f, "2,bob,40").unwrap();
    writeln!(f, "3,carol,25").unwrap();
    drop(f);

    let plan = plume_core::DataFrame::scan_csv(&path, schema())
        .filter(col("age").gt_eq(lit(30_i32))).unwrap()
        .project(vec![col("name")]).unwrap()
        .into_plan();

    let batches = collect(&plan).unwrap();
    let names: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_string::<i32>()
                .iter()
                .map(|o| o.unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(names, vec!["alice", "bob"]);
}

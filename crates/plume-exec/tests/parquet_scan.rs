//! Unit tests for ParquetScanExec — round-trip via the parquet writer.

use std::fs::File;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use plume_exec::{ExecutionPlan, ParquetScanExec};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

fn write_fixture(path: &std::path::Path, rows: usize) {
    let f = File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(f, schema(), None).unwrap();
    let ids: Vec<i64> = (0..rows as i64).collect();
    let prices: Vec<f64> = (0..rows).map(|i| i as f64 * 0.5).collect();
    let batch = RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int64Array::from(ids)), Arc::new(Float64Array::from(prices))],
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

#[test]
fn parquet_scan_streams_back_what_was_written() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("toy.parquet");
    write_fixture(&path, 100);

    let exec: Arc<dyn ExecutionPlan> = Arc::new(ParquetScanExec::new(path, schema()));
    let batches: Vec<RecordBatch> = exec
        .execute()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 100);
    let first = &batches[0];
    assert_eq!(first.column(0).as_primitive::<Int64Type>().value(0), 0);
    assert!((first.column(1).as_primitive::<Float64Type>().value(0) - 0.0).abs() < 1e-12);
}

#[test]
fn parquet_scan_handles_multi_row_group_file() {
    use parquet::file::properties::WriterProperties;

    // Force small row groups so the scan iterates more than one.
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rg.parquet");
    let f = File::create(&path).unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_size(50)
        .build();
    let mut w = ArrowWriter::try_new(f, schema(), Some(props)).unwrap();
    for chunk_start in (0..200_i64).step_by(50) {
        let ids: Vec<i64> = (chunk_start..chunk_start + 50).collect();
        let prices: Vec<f64> = (0..50).map(|_| 1.0).collect();
        let batch = RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int64Array::from(ids)), Arc::new(Float64Array::from(prices))],
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.flush().unwrap();
    }
    w.close().unwrap();

    let exec: Arc<dyn ExecutionPlan> = Arc::new(ParquetScanExec::new(path, schema()));
    let total: usize = exec
        .execute()
        .unwrap()
        .map(|r| r.unwrap().num_rows())
        .sum();
    assert_eq!(total, 200);
}

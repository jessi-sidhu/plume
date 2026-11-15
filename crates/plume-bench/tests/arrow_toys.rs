//! Week-1 "did I learn Arrow?" tests. Each one is a tiny end-to-end exercise
//! using the Arrow APIs that the engine will lean on later.

use std::fs;
use std::sync::Arc;

use anyhow::Result;
use arrow::compute::{kernels, sum};
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

fn tmp_path(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("plume-arrow-toys");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn build_filter_and_sum_int32() -> Result<()> {
    // Build a RecordBatch by hand, filter rows where col > 5, sum what remains.
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
    let arr = Int32Array::from(vec![1, 6, 3, 8, 5, 7, 10]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])?;

    // arrow 54 cmp kernels take &dyn Datum on both sides. ArrayRef coerces
    // for the column; new_scalar wraps the literal.
    let predicate = kernels::cmp::gt(batch.column(0), &Int32Array::new_scalar(5))?;
    let filtered = arrow::compute::filter_record_batch(&batch, &predicate)?;

    let total = sum(filtered.column(0).as_primitive::<Int32Type>()).unwrap();
    assert_eq!(total, 6 + 8 + 7 + 10);
    Ok(())
}

#[test]
fn write_then_read_parquet_roundtrip() -> Result<()> {
    use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};

    let path = tmp_path("toy.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )?;

    {
        let f = std::fs::File::create(&path)?;
        let mut w = ArrowWriter::try_new(f, schema.clone(), None)?;
        w.write(&batch)?;
        w.close()?;
    }

    let f = std::fs::File::open(&path)?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(f)?.build()?;
    let read = reader.next().unwrap()?;
    assert_eq!(read.num_rows(), 3);
    assert_eq!(read.column(1).as_string::<i32>().value(2), "c");
    Ok(())
}

#[test]
fn read_csv_via_arrow_csv_reader() -> Result<()> {
    use arrow::csv::ReaderBuilder;

    // Hand-rolled tiny CSV — the goal is to exercise the CSV reader API,
    // not to depend on the generated fixtures (which live behind gen_fixtures).
    let path = tmp_path("toy.csv");
    fs::write(&path, "id,name\n1,a\n2,b\n3,c\n")?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let f = std::fs::File::open(&path)?;
    let mut reader = ReaderBuilder::new(schema)
        .with_header(true)
        .with_batch_size(1024)
        .build(f)?;
    let batch = reader.next().unwrap()?;
    assert_eq!(batch.num_rows(), 3);
    Ok(())
}

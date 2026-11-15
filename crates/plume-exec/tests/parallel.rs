//! Tests for ParallelParquetScanExec — Week 13.
//!
//! Correctness tests compare against the sequential ParquetScanExec on
//! identical inputs. Output order is unspecified for the parallel path,
//! so we sort both sides before comparing.

use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plume_core::{col, lit};
use plume_exec::{ExecutionPlan, ParallelParquetScanExec, ParquetScanExec};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

/// Write a parquet file with `n_groups` row groups of `rows_per_group` rows.
/// Group `g` covers ids [g*rows_per_group, (g+1)*rows_per_group) and prices
/// in [g*100.0, (g+1)*100.0) so that pushdown can prune deterministically.
fn write_banded_parquet(path: &std::path::Path, n_groups: usize, rows_per_group: usize) {
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_size(rows_per_group)
        .build();
    let mut w = ArrowWriter::try_new(file, schema(), Some(props)).unwrap();
    for g in 0..n_groups {
        let start = (g * rows_per_group) as i64;
        let ids: Vec<i64> = (start..start + rows_per_group as i64).collect();
        let prices: Vec<f64> = (0..rows_per_group)
            .map(|i| g as f64 * 100.0 + (i as f64) * (100.0 / rows_per_group as f64))
            .collect();
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Float64Array::from(prices)),
            ],
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.flush().unwrap();
    }
    w.close().unwrap();
}

fn collect_ids(exec: Arc<dyn ExecutionPlan>) -> HashSet<i64> {
    exec.execute()
        .unwrap()
        .flat_map(|b| {
            let b = b.unwrap();
            (0..b.num_rows())
                .map(|i| b.column(0).as_primitive::<Int64Type>().value(i))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn collect_prices(exec: Arc<dyn ExecutionPlan>) -> Vec<f64> {
    let mut prices: Vec<f64> = exec
        .execute()
        .unwrap()
        .flat_map(|b| {
            let b = b.unwrap();
            (0..b.num_rows())
                .map(|i| b.column(1).as_primitive::<Float64Type>().value(i))
                .collect::<Vec<_>>()
        })
        .collect();
    prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
    prices
}

#[test]
fn parallel_full_scan_returns_same_id_set_as_sequential() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    write_banded_parquet(&path, 8, 1000);

    let seq: Arc<dyn ExecutionPlan> = Arc::new(ParquetScanExec::new(path.clone(), schema()));
    let par: Arc<dyn ExecutionPlan> =
        Arc::new(ParallelParquetScanExec::new(path.clone(), schema()));

    let seq_ids = collect_ids(seq);
    let par_ids = collect_ids(par);
    assert_eq!(
        seq_ids, par_ids,
        "parallel and sequential scans must see the same rows"
    );
    assert_eq!(par_ids.len(), 8000);
}

#[test]
fn parallel_scan_honours_pushdown_predicates() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    write_banded_parquet(&path, 8, 1000);

    // price > 700 leaves only groups 7 (prices in [700,800)) — wait, layout:
    // group g covers [g*100, (g+1)*100). So price > 700 leaves groups 7
    // (prices in [700,800)). 1 group * 1000 rows = 1000 surviving rows.
    let predicate = col("price").gt(lit(700.0_f64));
    let par: Arc<dyn ExecutionPlan> = Arc::new(
        ParallelParquetScanExec::new(path.clone(), schema())
            .with_pushdown_predicates(vec![predicate]),
    );
    let prices = collect_prices(par);
    assert_eq!(
        prices.len(),
        1000,
        "expected one row group of 1000 to survive"
    );
    // Every surviving price is from the [700, 800) band.
    assert!(
        prices.iter().all(|p| *p >= 700.0 && *p < 800.0),
        "out-of-band price detected: {:?}",
        &prices[..5]
    );
}

#[test]
fn parallel_scan_with_one_worker_matches_sequential() {
    // Pin to a single worker — the implementation must still be correct
    // when rayon's pool can't actually parallelise.
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    write_banded_parquet(&path, 4, 500);

    let seq: Arc<dyn ExecutionPlan> = Arc::new(ParquetScanExec::new(path.clone(), schema()));
    let par: Arc<dyn ExecutionPlan> =
        Arc::new(ParallelParquetScanExec::new(path.clone(), schema()).with_max_workers(1));
    assert_eq!(collect_ids(seq), collect_ids(par));
}

#[test]
fn parallel_scan_works_with_projection() {
    // Project only `price`; output schema must be one column, every row
    // present.
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    write_banded_parquet(&path, 4, 500);

    let projected_schema = Arc::new(Schema::new(vec![Field::new(
        "price",
        DataType::Float64,
        false,
    )]));
    let par: Arc<dyn ExecutionPlan> = Arc::new(ParallelParquetScanExec::with_projection(
        path.clone(),
        projected_schema.clone(),
        vec!["price".to_string()],
    ));

    let mut total_rows = 0;
    let mut saw_one_col = false;
    for batch in par.execute().unwrap() {
        let b = batch.unwrap();
        total_rows += b.num_rows();
        if b.num_columns() == 1 {
            saw_one_col = true;
        }
    }
    assert_eq!(total_rows, 2000);
    assert!(saw_one_col);
}

#[test]
fn parallel_scan_on_real_sales_fixture_matches_row_count() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures/sales.parquet");
    if !p.exists() {
        eprintln!("SKIP: {} missing — run gen_fixtures first", p.display());
        return;
    }
    let sales_schema: SchemaRef = plume_io::read_parquet_schema(&p).unwrap();

    let par: Arc<dyn ExecutionPlan> = Arc::new(ParallelParquetScanExec::new(
        p.clone(),
        sales_schema.clone(),
    ));
    let total: usize = par.execute().unwrap().map(|b| b.unwrap().num_rows()).sum();
    assert_eq!(total, 1_000_000);
}

//! Generate the two Week-1 fixtures into ./fixtures.
//!
//! - people.csv: 10K rows. Mixed types, sprinkled nulls. Used by the trivial
//!   harness tests and most early operator tests.
//!
//! - sales.parquet: 1M rows, 8 row groups of 125K rows each. Sorted by `price`
//!   *within* each row group, with each row group covering a different price
//!   band. This is deliberate: Week-9 row-group pruning needs row groups whose
//!   min/max statistics actually skew, otherwise the pruning demo is a no-op.
//!
//! Run from the workspace root:
//!     cargo run -p plume-bench --bin gen_fixtures

use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{
    Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const PEOPLE_ROWS: usize = 10_000;
const SALES_TOTAL: usize = 1_000_000;
const SALES_ROW_GROUPS: usize = 8;
const SALES_RG_ROWS: usize = SALES_TOTAL / SALES_ROW_GROUPS; // 125,000

fn fixtures_dir() -> Result<PathBuf> {
    // CARGO_MANIFEST_DIR points at crates/plume-bench at compile time, so
    // walk up to the workspace root regardless of cwd at runtime.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let root = PathBuf::from(manifest)
        .parent()
        .and_then(|p| p.parent())
        .context("locate workspace root from CARGO_MANIFEST_DIR")?
        .to_path_buf();
    let dir = root.join("fixtures");
    fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    Ok(dir)
}

fn write_people_csv(path: &PathBuf) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let names = ["alice", "bob", "carol", "dan", "eve", "frank", "grace", "heidi"];
    let regions = ["west", "east", "north", "south"];

    let mut s = String::new();
    s.push_str("id,name,age,region,score\n");
    for i in 0..PEOPLE_ROWS {
        let name = names[rng.gen_range(0..names.len())];
        // ~5% null ages, ~3% null scores — give 3-valued logic something to bite on.
        let age = if rng.gen_bool(0.05) {
            String::new()
        } else {
            rng.gen_range(0..90).to_string()
        };
        let region = regions[rng.gen_range(0..regions.len())];
        let score = if rng.gen_bool(0.03) {
            String::new()
        } else {
            format!("{:.4}", rng.gen_range(0.0_f64..100.0_f64))
        };
        s.push_str(&format!("{i},{name},{age},{region},{score}\n"));
    }
    fs::write(path, s).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn write_sales_parquet(path: &PathBuf) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
    ]));

    let categories = ["books", "electronics", "garden", "kitchen", "toys"];
    let regions = ["west", "east", "north", "south"];

    // 8 row groups, each covering its own price band:
    //   rg 0 -> [0, 125), rg 1 -> [125, 250), ... rg 7 -> [875, 1000)
    // Sorted ascending within each row group so min/max stats are tight.
    let props = WriterProperties::builder()
        .set_max_row_group_size(SALES_RG_ROWS)
        .set_write_batch_size(8192)
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    let mut order_id: i64 = 0;
    for rg in 0..SALES_ROW_GROUPS {
        let mut rng = StdRng::seed_from_u64(0xBEEF + rg as u64);
        let band_lo = (rg as f64) * 125.0;
        let band_hi = band_lo + 125.0;

        let mut prices: Vec<f64> = (0..SALES_RG_ROWS)
            .map(|_| rng.gen_range(band_lo..band_hi))
            .collect();
        prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let order_ids: Vec<i64> = (0..SALES_RG_ROWS)
            .map(|_| {
                order_id += 1;
                order_id
            })
            .collect();
        let categories_col: Vec<&str> = (0..SALES_RG_ROWS)
            .map(|_| categories[rng.gen_range(0..categories.len())])
            .collect();
        let regions_col: Vec<&str> = (0..SALES_RG_ROWS)
            .map(|_| regions[rng.gen_range(0..regions.len())])
            .collect();
        let qty: Vec<i32> = (0..SALES_RG_ROWS).map(|_| rng.gen_range(1..20)).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(order_ids)),
                Arc::new(StringArray::from(categories_col)),
                Arc::new(StringArray::from(regions_col)),
                Arc::new(Float64Array::from(prices)),
                Arc::new(Int32Array::from(qty)),
            ],
        )?;
        writer.write(&batch)?;
        // Force the row-group boundary to land exactly where we want it.
        writer.flush()?;
    }
    writer.close()?;
    Ok(())
}

fn main() -> Result<()> {
    let dir = fixtures_dir()?;
    let people = dir.join("people.csv");
    let sales = dir.join("sales.parquet");

    println!("writing {}", people.display());
    write_people_csv(&people)?;
    println!("writing {} ({SALES_TOTAL} rows, {SALES_ROW_GROUPS} row groups)", sales.display());
    write_sales_parquet(&sales)?;
    println!("done");
    Ok(())
}

//! plume-io: I/O helpers that bridge file formats and the DataFrame builder.
//!
//! For Week 7 the only public surface is `scan_parquet_auto`, which reads the
//! schema from a Parquet file's metadata so the caller doesn't have to spell
//! it out by hand. CSV scans still require an explicit schema (CSV has no
//! native type info; inference would be a separate concern).

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use arrow_schema::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use plume_core::DataFrame;

/// Read just the schema from a Parquet file (does not stream any data).
pub fn read_parquet_schema(path: impl AsRef<Path>) -> Result<SchemaRef> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("read parquet metadata for {}", path.display()))?;
    Ok(builder.schema().clone())
}

/// Build a `DataFrame` rooted at a `Parquet` scan, inferring schema from the
/// file's metadata. The caller still chains the rest of the query
/// (`.filter`, `.project`, `.aggregate`, ...).
pub fn scan_parquet_auto(path: impl AsRef<Path>) -> Result<DataFrame> {
    let schema = read_parquet_schema(&path)?;
    Ok(DataFrame::scan_parquet(path.as_ref().to_path_buf(), schema))
}

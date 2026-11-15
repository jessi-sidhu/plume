//! Week-1 correctness harness: diff a RecordBatch produced by plume against
//! the result DuckDB returns for an equivalent SQL query.
//!
//! The signature is intentionally simple: callers hand us *what they got* and
//! the SQL DuckDB should run for the same answer. Equality is checked by
//! converting both sides to a canonical row-of-strings form, which side-steps
//! float-NaN, dictionary encoding, and schema-metadata noise. Order matters by
//! default — pass `order_sensitive = false` if your query has no ORDER BY.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
    UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use arrow_schema::Schema;
use duckdb::Connection;
use parquet::arrow::ArrowWriter;

/// Pretty-print a value at row `i` of an Arrow array, using the marker
/// `<NULL>` for null cells. Kept narrow on purpose — extend as new types
/// appear in tests.
fn render_cell(array: &dyn Array, i: usize) -> Result<String> {
    if array.is_null(i) {
        return Ok("<NULL>".to_string());
    }
    Ok(match array.data_type() {
        DataType::Boolean => array.as_boolean().value(i).to_string(),
        DataType::Int8 => array.as_primitive::<Int8Type>().value(i).to_string(),
        DataType::Int16 => array.as_primitive::<Int16Type>().value(i).to_string(),
        DataType::Int32 => array.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => array.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::UInt8 => array.as_primitive::<UInt8Type>().value(i).to_string(),
        DataType::UInt16 => array.as_primitive::<UInt16Type>().value(i).to_string(),
        DataType::UInt32 => array.as_primitive::<UInt32Type>().value(i).to_string(),
        DataType::UInt64 => array.as_primitive::<UInt64Type>().value(i).to_string(),
        // Floats: scientific with ~10 significant digits. Fixed-decimal
        // formats penalize whichever side is bigger (a 4-decimal AVG of 500
        // resolves to ~7 sig figs but the same 4 decimals of 0.0007 is just
        // 1 sig fig). Sig-figs is the only fair scale across magnitudes,
        // and 10 sig figs survives 1M-row SUM/AVG drift between engines.
        DataType::Float32 => format!("{:.5e}", array.as_primitive::<Float32Type>().value(i)),
        DataType::Float64 => format!("{:.9e}", array.as_primitive::<Float64Type>().value(i)),
        DataType::Utf8 => array.as_string::<i32>().value(i).to_string(),
        DataType::LargeUtf8 => array.as_string::<i64>().value(i).to_string(),
        // DuckDB returns SUM(integer) as DECIMAL(38,0) by default; render as
        // the underlying i128 so it compares cleanly against plume's Int64.
        DataType::Decimal128(_, scale) => {
            let v = array.as_primitive::<Decimal128Type>().value(i);
            if *scale == 0 {
                v.to_string()
            } else {
                format!("decimal128(scale={scale},val={v})")
            }
        }
        other => bail!("render_cell: unsupported type {other:?}"),
    })
}

fn batch_to_rows(batch: &RecordBatch) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::with_capacity(batch.num_rows());
    for r in 0..batch.num_rows() {
        let mut row = Vec::with_capacity(batch.num_columns());
        for c in 0..batch.num_columns() {
            row.push(render_cell(batch.column(c).as_ref(), r)?);
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Concatenate a slice of RecordBatches (assumed same schema) into rows.
fn batches_to_rows(batches: &[RecordBatch]) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    for b in batches {
        rows.extend(batch_to_rows(b)?);
    }
    Ok(rows)
}

/// Run `sql` in an embedded DuckDB and return the result as RecordBatches.
/// `register_csvs` and `register_parquets` are (table_name, path) pairs that
/// are made available to the query as views over the file.
pub fn duckdb_query(
    sql: &str,
    register_csvs: &[(&str, &Path)],
    register_parquets: &[(&str, &Path)],
) -> Result<Vec<RecordBatch>> {
    let conn = Connection::open_in_memory().context("open duckdb in-memory")?;
    for (name, path) in register_csvs {
        let p = path.to_str().ok_or_else(|| anyhow!("non-utf8 csv path"))?;
        conn.execute_batch(&format!(
            "CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{p}');"
        ))
        .with_context(|| format!("register csv view {name}"))?;
    }
    for (name, path) in register_parquets {
        let p = path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 parquet path"))?;
        conn.execute_batch(&format!(
            "CREATE VIEW {name} AS SELECT * FROM read_parquet('{p}');"
        ))
        .with_context(|| format!("register parquet view {name}"))?;
    }
    let mut stmt = conn.prepare(sql).context("prepare sql")?;
    let batches: Vec<RecordBatch> = stmt.query_arrow([]).context("execute sql")?.collect();
    Ok(batches)
}

/// Diff `actual` (whatever plume produced) against the result DuckDB returns
/// for `sql`. Panics with a readable message on mismatch.
pub fn assert_matches_duckdb(
    actual: &[RecordBatch],
    sql: &str,
    register_csvs: &[(&str, &Path)],
    register_parquets: &[(&str, &Path)],
    order_sensitive: bool,
) -> Result<()> {
    let expected = duckdb_query(sql, register_csvs, register_parquets)
        .with_context(|| format!("running reference SQL: {sql}"))?;

    let mut actual_rows = batches_to_rows(actual).context("rendering actual")?;
    let mut expected_rows = batches_to_rows(&expected).context("rendering expected")?;

    if !order_sensitive {
        actual_rows.sort();
        expected_rows.sort();
    }

    if actual_rows != expected_rows {
        // Cap the printed diff so a 1M-row mismatch doesn't blow up the test log.
        const SHOW: usize = 20;
        let a = actual_rows.iter().take(SHOW).collect::<Vec<_>>();
        let e = expected_rows.iter().take(SHOW).collect::<Vec<_>>();
        bail!(
            "result mismatch for SQL `{sql}`\n  actual rows ({} total, first {SHOW}):\n{:#?}\n  expected rows ({} total, first {SHOW}):\n{:#?}",
            actual_rows.len(),
            a,
            expected_rows.len(),
            e,
        );
    }
    Ok(())
}

/// Write a single batch to a fresh Parquet file inside `dir`. Returns the
/// path. Used by Week-3 evaluator tests that need DuckDB to see the same
/// input the plume evaluator just consumed.
pub fn write_batch_parquet(dir: &Path, name: &str, batch: &RecordBatch) -> Result<PathBuf> {
    let path = dir.join(format!("{name}.parquet"));
    let file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
    let schema: Arc<Schema> = batch.schema();
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(batch)?;
    writer.close()?;
    Ok(path)
}

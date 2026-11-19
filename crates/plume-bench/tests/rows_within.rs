//! `assert_rows_within`: every returned row must be in the reference
//! result, counting duplicates (like `EXCEPT ALL` leaving nothing).

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use plume_bench::harness::{assert_rows_within, reference_connection};

fn rows(k: Vec<Option<i64>>, s: Vec<Option<&str>>) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    vec![RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(k)),
            Arc::new(StringArray::from(s)),
        ],
    )
    .unwrap()]
}

/// (1,'a'), (1,'a'), (2,NULL), (NULL,'b')
const REFERENCE: &str =
    "SELECT * FROM (VALUES (1::BIGINT, 'a'), (1, 'a'), (2, NULL), (NULL, 'b')) AS t(k, s)";

#[test]
fn rows_inside_the_reference_pass() -> anyhow::Result<()> {
    let conn = reference_connection()?;
    let actual = rows(
        vec![Some(1), Some(2), None, Some(1)],
        vec![Some("a"), None, Some("b"), Some("a")],
    );
    assert_rows_within(&conn, &actual, REFERENCE)?;
    assert_rows_within(&conn, &[], REFERENCE)?;
    Ok(())
}

#[test]
fn a_row_not_in_the_reference_fails() -> anyhow::Result<()> {
    let conn = reference_connection()?;
    let actual = rows(vec![Some(1), Some(3)], vec![Some("a"), Some("a")]);
    let err = assert_rows_within(&conn, &actual, REFERENCE).unwrap_err();
    assert!(err.to_string().starts_with("1 returned rows"), "{err:#}");
    // NULL must match NULL, and only NULL.
    let actual = rows(vec![Some(2)], vec![Some("")]);
    assert!(assert_rows_within(&conn, &actual, REFERENCE).is_err());
    Ok(())
}

#[test]
fn duplicates_count() -> anyhow::Result<()> {
    let conn = reference_connection()?;
    // The reference has (1,'a') twice and (2,NULL) once.
    let actual = rows(
        vec![Some(1), Some(1), Some(1), Some(2), Some(2)],
        vec![Some("a"), Some("a"), Some("a"), None, None],
    );
    let err = assert_rows_within(&conn, &actual, REFERENCE).unwrap_err();
    assert!(err.to_string().starts_with("2 returned rows"), "{err:#}");
    Ok(())
}

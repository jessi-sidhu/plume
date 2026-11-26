//! Direct tests for the compiled-expression module — Week 11.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{Array, BooleanArray, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use plume_core::{col, lit, not};
use plume_exec::{compile, EvalError};

fn batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(2),
                None,
                Some(4),
                Some(5),
            ])),
            Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5, 4.5, 5.5])),
        ],
    )
    .unwrap()
}

#[test]
fn compile_resolves_column_to_index() {
    let b = batch();
    let c = compile(&col("b"), b.schema().as_ref()).unwrap();
    let arr = c.evaluate(&b).unwrap();
    let f = arr.as_primitive::<Float64Type>();
    assert_eq!(f.value(0), 1.5);
    assert_eq!(f.value(4), 5.5);
}

#[test]
fn compile_strips_alias() {
    // Alias is cosmetic; compile() should produce the same evaluator as the
    // unaliased expression.
    let b = batch();
    let c = compile(&col("a").alias("renamed"), b.schema().as_ref()).unwrap();
    let arr = c.evaluate(&b).unwrap();
    assert_eq!(arr.as_primitive::<Int32Type>().value(0), 1);
}

#[test]
fn compile_rejects_missing_column_at_compile_time() {
    let b = batch();
    let err = compile(&col("nope"), b.schema().as_ref()).unwrap_err();
    assert!(matches!(err, EvalError::MissingColumn(s) if s == "nope"));
}

#[test]
fn compile_rejects_aggregates_at_compile_time() {
    let b = batch();
    let err = compile(&plume_core::sum(col("a")), b.schema().as_ref()).unwrap_err();
    assert!(matches!(err, EvalError::AggregateInScalarContext));
}

#[test]
fn compiled_predicate_matches_evaluator_with_nulls() {
    let b = batch();
    let pred = col("a").gt(lit(2_i32));
    let c = compile(&pred, b.schema().as_ref()).unwrap();
    let mask = c.evaluate_predicate(&b).unwrap();
    let expected = BooleanArray::from(vec![Some(false), Some(false), None, Some(true), Some(true)]);
    assert_eq!(mask, expected);
}

#[test]
fn compiled_binary_op_matches_unoptimized_path() {
    // (a > 1) AND (b < 5.0) — chains through the compiled path.
    let b = batch();
    let pred = col("a").gt(lit(1_i32)).and(col("b").lt(lit(5.0_f64)));
    let c = compile(&pred, b.schema().as_ref()).unwrap();
    let mask = c.evaluate_predicate(&b).unwrap();
    // a>1: [F,T,N,T,T];  b<5.0: [T,T,T,T,F]
    // AND: [F,T,N,T,F]
    let expected = BooleanArray::from(vec![Some(false), Some(true), None, Some(true), Some(false)]);
    assert_eq!(mask, expected);
}

#[test]
fn literal_broadcasts_to_batch_length() {
    let b = batch();
    let c = compile(&lit(42_i32), b.schema().as_ref()).unwrap();
    let arr = c.evaluate(&b).unwrap();
    assert_eq!(arr.len(), b.num_rows());
    assert_eq!(arr.as_primitive::<Int32Type>().value(0), 42);
    assert_eq!(arr.as_primitive::<Int32Type>().value(4), 42);
}

#[test]
fn compiled_not_and_null_checks_match_interpreter() {
    let b = batch();
    let exprs = [
        not(col("a").gt(lit(2_i32))),
        col("a").is_null(),
        col("a").is_not_null(),
        not(col("a").is_null()),
    ];
    for e in exprs {
        let c = compile(&e, b.schema().as_ref()).unwrap();
        assert_eq!(c.data_type(), &DataType::Boolean, "{}", e.display_name());
        let got = c.evaluate(&b).unwrap();
        let want = plume_exec::evaluate(&e, &b).unwrap();
        assert_eq!(got.as_boolean(), want.as_boolean(), "{}", e.display_name());
    }
}

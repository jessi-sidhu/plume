//! Direct (non-harness) tests for the expression evaluator. Fast feedback
//! for the hot path; semantic-equivalence vs DuckDB lives in plume-bench.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use plume_core::{col, lit, Expr, ScalarValue};
use plume_exec::{evaluate, evaluate_predicate, EvalError};

fn ints_with_nulls() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(2), None, Some(4), Some(5)])),
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30), Some(40), Some(5)])),
        ],
    )
    .unwrap()
}

fn bools_for_kleene() -> RecordBatch {
    // Every (Boolean, Boolean) pair from the truth table — tt, tf, tn, ft, ff,
    // fn, nt, nf, nn — used to verify and_kleene / or_kleene line up with SQL.
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Boolean, true),
        Field::new("b", DataType::Boolean, true),
    ]));
    let a = vec![Some(true), Some(true), None, Some(false), Some(false), None, None, Some(true), Some(false)];
    let b = vec![Some(true), Some(false), Some(true), Some(true), Some(false), Some(false), None, None, None];
    RecordBatch::try_new(
        schema,
        vec![Arc::new(BooleanArray::from(a)), Arc::new(BooleanArray::from(b))],
    )
    .unwrap()
}

#[test]
fn column_reference_returns_underlying_array() {
    let batch = ints_with_nulls();
    let out = evaluate(&col("a"), &batch).unwrap();
    let a: &Int32Array = out.as_primitive::<Int32Type>();
    assert_eq!(a.len(), 5);
    assert_eq!(a.value(0), 1);
    assert!(a.is_null(2));
}

#[test]
fn missing_column_is_an_error() {
    let batch = ints_with_nulls();
    let err = evaluate(&col("nope"), &batch).unwrap_err();
    assert!(matches!(err, EvalError::MissingColumn(s) if s == "nope"));
}

#[test]
fn literal_materializes_to_full_length_array() {
    let batch = ints_with_nulls();
    for (e, expected_ty) in [
        (lit(7_i32), DataType::Int32),
        (lit(7_i64), DataType::Int64),
        (lit(7.5_f64), DataType::Float64),
        (lit(true), DataType::Boolean),
        (lit("hello"), DataType::Utf8),
    ] {
        let out = evaluate(&e, &batch).unwrap();
        assert_eq!(out.len(), batch.num_rows());
        assert_eq!(out.data_type(), &expected_ty);
    }
}

#[test]
fn untyped_null_literal_is_rejected() {
    let batch = ints_with_nulls();
    let err = evaluate(&Expr::Literal(ScalarValue::Null), &batch).unwrap_err();
    assert!(matches!(err, EvalError::UntypedNullLiteral));
}

#[test]
fn aggregate_in_scalar_context_is_rejected() {
    let batch = ints_with_nulls();
    let err = evaluate(&plume_core::sum(col("a")), &batch).unwrap_err();
    assert!(matches!(err, EvalError::AggregateInScalarContext));
}

#[test]
fn comparison_propagates_null_per_three_valued_logic() {
    let batch = ints_with_nulls();
    // a > 2 over [1, 2, NULL, 4, 5] -> [false, false, NULL, true, true]
    let out = evaluate_predicate(&col("a").gt(lit(2_i32)), &batch).unwrap();
    let expected = BooleanArray::from(vec![Some(false), Some(false), None, Some(true), Some(true)]);
    assert_eq!(&out, &expected);
}

#[test]
fn equality_handles_nulls_on_both_sides() {
    let batch = ints_with_nulls();
    // a = b: [1=10, 2=NULL, NULL=30, 4=40, 5=5] -> [false, NULL, NULL, false, true]
    let out = evaluate_predicate(&col("a").eq(col("b")), &batch).unwrap();
    let expected = BooleanArray::from(vec![Some(false), None, None, Some(false), Some(true)]);
    assert_eq!(&out, &expected);
}

#[test]
fn arithmetic_add_sub_mul_div_on_int32() {
    let batch = ints_with_nulls();
    // a + 1 over [1, 2, NULL, 4, 5] -> [2, 3, NULL, 5, 6]
    let out = evaluate(&col("a").plus(lit(1_i32)), &batch).unwrap();
    let arr = out.as_primitive::<Int32Type>();
    assert_eq!(arr.value(0), 2);
    assert!(arr.is_null(2));
    assert_eq!(arr.value(4), 6);

    // a * 10 -> [10, 20, NULL, 40, 50]
    let out = evaluate(&col("a").mul(lit(10_i32)), &batch).unwrap();
    assert_eq!(out.as_primitive::<Int32Type>().value(3), 40);

    // a - b -> [-9, NULL, NULL, -36, 0]
    let out = evaluate(&col("a").minus(col("b")), &batch).unwrap();
    let arr = out.as_primitive::<Int32Type>();
    assert_eq!(arr.value(0), -9);
    assert!(arr.is_null(1));
    assert_eq!(arr.value(4), 0);

    // b / a -> [10, NULL, NULL, 10, 1]   (integer division)
    let out = evaluate(&col("b").div(col("a")), &batch).unwrap();
    let arr = out.as_primitive::<Int32Type>();
    assert_eq!(arr.value(0), 10);
    assert!(arr.is_null(2));
    assert_eq!(arr.value(4), 1);
}

#[test]
fn arithmetic_works_on_float64() {
    let schema = Arc::new(Schema::new(vec![Field::new("p", DataType::Float64, false)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Float64Array::from(vec![1.5, 2.5, 4.0]))],
    )
    .unwrap();
    let out = evaluate(&col("p").mul(lit(2.0_f64)), &batch).unwrap();
    let arr = out.as_primitive::<Float64Type>();
    assert!((arr.value(0) - 3.0).abs() < 1e-12);
    assert!((arr.value(2) - 8.0).abs() < 1e-12);
}

#[test]
fn arithmetic_promotes_int32_into_int64_when_kernel_supports_it() {
    // Ensures our wrapper doesn't accidentally lose info. Int32 + Int32 should
    // stay Int32 (kernel preserves type); the SUM aggregate is what promotes.
    let batch = ints_with_nulls();
    let out = evaluate(&col("a").plus(col("b")), &batch).unwrap();
    assert_eq!(out.data_type(), &DataType::Int32);
    let _ = out.as_primitive::<Int32Type>();
}

#[test]
fn and_kleene_truth_table_matches_sql() {
    // bools_for_kleene rows in (a, b) order are:
    //   (T,T) (T,F) (N,T) (F,T) (F,F) (N,F) (N,N) (T,N) (F,N)
    // Kleene AND (NULL absorbs unless one side is false):
    //    T     F     N     F     F     F     N     N     F
    let batch = bools_for_kleene();
    let out = evaluate_predicate(&col("a").and(col("b")), &batch).unwrap();
    let expected = BooleanArray::from(vec![
        Some(true), Some(false), None, Some(false), Some(false), Some(false), None, None, Some(false),
    ]);
    assert_eq!(&out, &expected);
}

#[test]
fn or_kleene_truth_table_matches_sql() {
    // bools_for_kleene rows: (T,T) (T,F) (N,T) (F,T) (F,F) (N,F) (N,N) (T,N) (F,N)
    // Kleene OR (NULL absorbs unless one side is true):
    //  T     T     T     T     F     N     N     T     N
    let batch = bools_for_kleene();
    let out = evaluate_predicate(&col("a").or(col("b")), &batch).unwrap();
    let expected = BooleanArray::from(vec![
        Some(true), Some(true), Some(true), Some(true), Some(false), None, None, Some(true), None,
    ]);
    assert_eq!(&out, &expected);
}

#[test]
fn alias_is_transparent_in_eval() {
    let batch = ints_with_nulls();
    let out = evaluate(&col("a").alias("renamed"), &batch).unwrap();
    let direct = evaluate(&col("a"), &batch).unwrap();
    assert_eq!(out.as_primitive::<Int32Type>(), direct.as_primitive::<Int32Type>());
}

#[test]
fn predicate_check_rejects_non_boolean_result() {
    let batch = ints_with_nulls();
    let err = evaluate_predicate(&col("a"), &batch).unwrap_err();
    assert!(matches!(err, EvalError::PredicateNotBoolean(DataType::Int32)));
}

#[test]
fn chained_predicate_with_nested_and_handles_nulls() {
    let batch = ints_with_nulls();
    // (a > 1) AND (b < 50)
    // a>1: [F,T,N,T,T];  b<50: [T,N,T,T,T]
    // AND: [F,N,N,T,T]
    let pred = col("a").gt(lit(1_i32)).and(col("b").lt(lit(50_i32)));
    let out = evaluate_predicate(&pred, &batch).unwrap();
    let expected = BooleanArray::from(vec![Some(false), None, None, Some(true), Some(true)]);
    assert_eq!(&out, &expected);
}

#[test]
fn string_equality_works() {
    let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec!["alice", "bob", "alice"]))],
    )
    .unwrap();
    let out = evaluate_predicate(&col("name").eq(lit("alice")), &batch).unwrap();
    let expected = BooleanArray::from(vec![Some(true), Some(false), Some(true)]);
    assert_eq!(&out, &expected);
}

#[test]
fn int64_round_trip_preserves_value() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![1_i64, 2, 3]))],
    )
    .unwrap();
    let out = evaluate(&col("id").plus(lit(100_i64)), &batch).unwrap();
    let arr = out.as_primitive::<Int64Type>();
    assert_eq!(arr.value(2), 103);
}

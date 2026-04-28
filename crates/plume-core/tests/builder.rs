//! Week-2 builder tests: explain output, schema propagation, error paths.
//! No execution involved — these test that the plan tree itself is correct.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use plume_core::expr::AggregateFunction;
use plume_core::{
    avg, col, count_star, lit, sum, DataFrame, Expr, LogicalPlan, PlumeError,
};

fn sales_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
    ]))
}

#[test]
fn explain_renders_chain_top_down() {
    let df = DataFrame::scan_parquet("sales.parquet", sales_schema())
        .filter(col("region").eq(lit("west"))).unwrap()
        .project(vec![col("category"), col("price")]).unwrap()
        .limit(10);

    let s = df.explain();
    let expected = "\
Limit: 10
  Projection: [category, price]
    Filter: (region = \"west\")
      ParquetScan: sales.parquet [5 cols]
";
    assert_eq!(s, expected, "got:\n{s}");
}

#[test]
fn projection_rewrites_schema_in_order() {
    let df = DataFrame::scan_csv("p.csv", sales_schema())
        .project(vec![col("price"), col("category").alias("cat")])
        .unwrap();
    let schema = df.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["price", "cat"]);
    assert_eq!(schema.field(0).data_type(), &DataType::Float64);
    assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
}

#[test]
fn filter_preserves_input_schema() {
    let in_schema = sales_schema();
    let df = DataFrame::scan_parquet("p.parquet", in_schema.clone())
        .filter(col("price").gt(lit(100.0_f64))).unwrap();
    assert_eq!(df.schema().fields(), in_schema.fields());
}

#[test]
fn aggregate_emits_group_then_agg_fields() {
    let df = DataFrame::scan_parquet("p.parquet", sales_schema())
        .aggregate(
            vec![col("region"), col("category")],
            vec![avg(col("price")).alias("avg_price"), count_star(), sum(col("quantity"))],
        )
        .unwrap();

    let schema = df.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["region", "category", "avg_price", "count_star()", "sum(quantity)"]);

    // Numeric promotion rules: SUM(Int32) -> Int64, AVG(any numeric) -> Float64,
    // COUNT(*) -> Int64.
    assert_eq!(schema.field(2).data_type(), &DataType::Float64);
    assert_eq!(schema.field(3).data_type(), &DataType::Int64);
    assert_eq!(schema.field(4).data_type(), &DataType::Int64);

    // count_star is non-nullable; aggregates over numeric are nullable.
    assert!(!schema.field(3).is_nullable());
    assert!(schema.field(2).is_nullable());
}

#[test]
fn missing_column_is_a_schema_error() {
    let err = DataFrame::scan_parquet("p.parquet", sales_schema())
        .filter(col("nope").eq(lit(1_i32)))
        .unwrap_err();
    matches!(err, PlumeError::Schema(_)).then_some(()).expect("expected schema error");
}

#[test]
fn non_boolean_filter_predicate_is_a_type_error() {
    let err = DataFrame::scan_parquet("p.parquet", sales_schema())
        .filter(col("price"))   // Float64, not Boolean
        .unwrap_err();
    matches!(err, PlumeError::Type(_)).then_some(()).expect("expected type error");
}

#[test]
fn comparison_with_mismatched_types_is_a_type_error() {
    let err = DataFrame::scan_parquet("p.parquet", sales_schema())
        .filter(col("price").eq(lit("not a float")))
        .unwrap_err();
    matches!(err, PlumeError::Type(_)).then_some(()).expect("expected type error");
}

#[test]
fn aggregate_rejects_non_aggregate_expr_in_agg_position() {
    let err = DataFrame::scan_parquet("p.parquet", sales_schema())
        .aggregate(vec![col("region")], vec![col("price")])  // col, not an agg
        .unwrap_err();
    matches!(err, PlumeError::Plan(_)).then_some(()).expect("expected plan error");
}

#[test]
fn alias_renames_output_field() {
    let df = DataFrame::scan_parquet("p.parquet", sales_schema())
        .project(vec![col("price").alias("usd")])
        .unwrap();
    assert_eq!(df.schema().field(0).name(), "usd");
    assert_eq!(df.schema().field(0).data_type(), &DataType::Float64);
}

#[test]
fn aggregate_function_helpers_construct_expected_variants() {
    // Light sanity check on the helper constructors.
    if let Expr::Aggregate { fun, .. } = avg(col("x")) {
        assert_eq!(fun, AggregateFunction::Avg);
    } else {
        panic!("avg() did not produce Aggregate variant");
    }
    if let Expr::Aggregate { fun, .. } = count_star() {
        assert_eq!(fun, AggregateFunction::CountStar);
    } else {
        panic!("count_star() did not produce Aggregate variant");
    }
}

#[test]
fn binary_op_resolves_to_boolean() {
    let s = sales_schema();
    let e = col("region").eq(lit("west"));
    assert_eq!(e.data_type(&s).unwrap(), DataType::Boolean);

    let chained = col("price").gt(lit(10.0_f64)).and(col("region").eq(lit("east")));
    assert_eq!(chained.data_type(&s).unwrap(), DataType::Boolean);
}

#[test]
fn into_plan_round_trip() {
    let df = DataFrame::scan_csv("p.csv", sales_schema()).limit(5);
    let plan: LogicalPlan = df.into_plan();
    assert_eq!(plan.schema().fields().len(), 5);
}

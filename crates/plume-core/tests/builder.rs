//! Week-2 builder tests: explain output, schema propagation, error paths.
//! No execution involved — these test that the plan tree itself is correct.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use plume_core::expr::AggregateFunction;
use plume_core::{
    avg, col, count_star, lit, sum, DataFrame, Expr, JoinType, LogicalPlan, PlumeError,
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

// =======================================================================
// Join builder tests
// =======================================================================

fn customers_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

fn orders_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),         // collides with customers.id
        Field::new("customer_id", DataType::Int64, false),
        Field::new("amount", DataType::Float64, false),
    ]))
}

#[test]
fn inner_join_concatenates_schemas_with_disambiguation() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let joined = left
        .join(
            right,
            vec![(col("id"), col("customer_id"))],
            JoinType::Inner,
        )
        .unwrap();

    let schema = joined.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    // customers.id stays as `id`; orders.id collides and becomes `id_1`.
    assert_eq!(names, vec!["id", "name", "id_1", "customer_id", "amount"]);
    // Inner join preserves input nullability — none of these were nullable.
    assert!(schema.fields().iter().all(|f| !f.is_nullable()));
}

#[test]
fn left_join_widens_right_side_nullability() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let joined = left
        .join(right, vec![(col("id"), col("customer_id"))], JoinType::Left)
        .unwrap();

    let schema = joined.schema();
    // Left columns: still non-nullable.
    assert!(!schema.field_with_name("id").unwrap().is_nullable());
    assert!(!schema.field_with_name("name").unwrap().is_nullable());
    // Right columns: widened to nullable because unmatched left rows
    // produce NULL on the right side.
    assert!(schema.field_with_name("id_1").unwrap().is_nullable());
    assert!(schema.field_with_name("customer_id").unwrap().is_nullable());
    assert!(schema.field_with_name("amount").unwrap().is_nullable());
}

#[test]
fn right_join_widens_left_side_nullability() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let joined = left
        .join(right, vec![(col("id"), col("customer_id"))], JoinType::Right)
        .unwrap();

    let schema = joined.schema();
    assert!(schema.field_with_name("id").unwrap().is_nullable());
    assert!(schema.field_with_name("name").unwrap().is_nullable());
    assert!(!schema.field_with_name("id_1").unwrap().is_nullable());
    assert!(!schema.field_with_name("customer_id").unwrap().is_nullable());
    assert!(!schema.field_with_name("amount").unwrap().is_nullable());
}

#[test]
fn full_join_widens_both_sides() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let joined = left
        .join(right, vec![(col("id"), col("customer_id"))], JoinType::Full)
        .unwrap();
    let schema = joined.schema();
    assert!(schema.fields().iter().all(|f| f.is_nullable()));
}

#[test]
fn semi_join_projects_only_left_schema() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let joined = left
        .join(right, vec![(col("id"), col("customer_id"))], JoinType::Semi)
        .unwrap();

    let schema = joined.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["id", "name"]);
    // Semi/Anti don't widen — they only emit left rows that *did* match.
    assert!(schema.fields().iter().all(|f| !f.is_nullable()));
}

#[test]
fn anti_join_projects_only_left_schema() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let joined = left
        .join(right, vec![(col("id"), col("customer_id"))], JoinType::Anti)
        .unwrap();
    let schema = joined.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["id", "name"]);
}

#[test]
fn cross_join_concatenates_without_widening() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let joined = left.cross_join(right).unwrap();
    let schema = joined.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["id", "name", "id_1", "customer_id", "amount"]);
    assert!(schema.fields().iter().all(|f| !f.is_nullable()));
}

#[test]
fn equi_key_type_mismatch_is_a_type_error() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    // customers.name is Utf8, orders.amount is Float64 — incomparable.
    let err = left
        .join(right, vec![(col("name"), col("amount"))], JoinType::Inner)
        .unwrap_err();
    matches!(err, PlumeError::Type(_)).then_some(()).expect("expected type error");
}

#[test]
fn inner_join_with_no_keys_is_a_plan_error() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let err = left.join(right, vec![], JoinType::Inner).unwrap_err();
    matches!(err, PlumeError::Plan(_)).then_some(()).expect("expected plan error");
}

#[test]
fn cross_join_with_keys_is_a_plan_error() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let err = left
        .join(right, vec![(col("id"), col("customer_id"))], JoinType::Cross)
        .unwrap_err();
    matches!(err, PlumeError::Plan(_)).then_some(()).expect("expected plan error");
}

#[test]
fn unknown_left_key_column_is_a_schema_error() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let err = left
        .join(right, vec![(col("nope"), col("customer_id"))], JoinType::Inner)
        .unwrap_err();
    matches!(err, PlumeError::Schema(_)).then_some(()).expect("expected schema error");
}

#[test]
fn unknown_right_key_column_is_a_schema_error() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let err = left
        .join(right, vec![(col("id"), col("nope"))], JoinType::Inner)
        .unwrap_err();
    matches!(err, PlumeError::Schema(_)).then_some(()).expect("expected schema error");
}

#[test]
fn join_explain_renders_two_arm_subtree() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let joined = left
        .join(right, vec![(col("id"), col("customer_id"))], JoinType::Inner)
        .unwrap();

    let s = joined.explain();
    let expected = "\
InnerJoin: on=[id = customer_id]
  L:
    ParquetScan: customers.parquet [2 cols]
  R:
    ParquetScan: orders.parquet [3 cols]
";
    assert_eq!(s, expected, "got:\n{s}");
}

#[test]
fn cross_join_explain_omits_on_clause() {
    let left = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let right = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let joined = left.cross_join(right).unwrap();
    let s = joined.explain();
    let expected = "\
CrossJoin
  L:
    ParquetScan: customers.parquet [2 cols]
  R:
    ParquetScan: orders.parquet [3 cols]
";
    assert_eq!(s, expected, "got:\n{s}");
}

#[test]
fn three_way_join_via_chained_builder() {
    // A chained join of three relations builds a left-deep tree:
    //   (customers ⋈ orders) ⋈ products
    let products_schema = Arc::new(Schema::new(vec![
        Field::new("product_id", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let customers = DataFrame::scan_parquet("customers.parquet", customers_schema());
    let orders = DataFrame::scan_parquet("orders.parquet", orders_schema());
    let products = DataFrame::scan_parquet("products.parquet", products_schema);

    // First join on customers.id = orders.customer_id.
    let two = customers
        .join(orders, vec![(col("id"), col("customer_id"))], JoinType::Inner)
        .unwrap();
    // Now join the result on (concatenated) id_1 against products.product_id.
    let three = two
        .join(
            products,
            vec![(col("id_1"), col("product_id"))],
            JoinType::Inner,
        )
        .unwrap();
    let schema = three.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["id", "name", "id_1", "customer_id", "amount", "product_id", "price"]);
}

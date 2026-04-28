//! Global and grouped aggregation.
//!
//! Each `Accumulator` owns the per-aggregate state. Supported set is
//! COUNT, COUNT(*), SUM, AVG, MIN, MAX over the primitive numeric types,
//! plus Utf8 MIN/MAX. Strict per-SQL null semantics:
//!
//! - COUNT(expr) skips nulls; COUNT(*) counts every row.
//! - SUM/AVG/MIN/MAX over an empty (or all-null) input return NULL.
//! - SUM(Int*) promotes to Int64; SUM(UInt*) to UInt64; SUM(Float*) to Float64.
//! - AVG always returns Float64.
//! - MIN/MAX preserve input type.
//!
//! Overflow follows Arrow's wrap-on-overflow behaviour for Int64/UInt64 sums,
//! which matches DuckDB's default. Documented for the README's limitations
//! section.
//!
//! Grouped aggregation (`HashAggregateExec`, Week 6) uses a row-at-a-time
//! probe-and-update loop with a `HashMap<Vec<KeyValue>, Vec<Box<Accumulator>>>`.
//! That's the simplest correct implementation. A vectorized group-by (hash
//! whole columns at once) is a known faster path; revisit in Week 11/13.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::{cast, concat};
use arrow::compute::kernels::aggregate;
use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int32Builder, Int64Builder, StringBuilder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type, UInt64Type};
use arrow_array::{
    new_null_array, Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray,
    UInt64Array,
};
use arrow_schema::{DataType, Schema, SchemaRef};
use plume_core::{AggregateFunction, Expr};

use crate::eval::{evaluate, EvalError, EvalResult};
use crate::exec::{BatchStream, ExecutionPlan};

// ------------------------------------------------------------------------
// Trait
// ------------------------------------------------------------------------

pub trait Accumulator: Send {
    fn data_type(&self) -> DataType;
    fn update_batch(&mut self, batch: &RecordBatch) -> EvalResult<()>;
    /// Produce a single-row array containing the final aggregate value.
    fn finalize(&mut self) -> EvalResult<ArrayRef>;
}

/// Build the right accumulator for `agg_expr` against `input_schema`.
/// `agg_expr` must be `Expr::Aggregate { .. }` (optionally wrapped in
/// `Expr::Alias`). Anything else is a planning bug.
pub fn create_accumulator(
    agg_expr: &Expr,
    input_schema: &Schema,
) -> EvalResult<Box<dyn Accumulator>> {
    let (fun, arg) = unwrap_aggregate(agg_expr)?;
    Ok(match fun {
        AggregateFunction::CountStar => Box::new(CountStarAccumulator::default()),
        AggregateFunction::Count => Box::new(CountAccumulator::new(arg)),
        AggregateFunction::Sum => Box::new(SumAccumulator::new(arg, input_schema)?),
        AggregateFunction::Avg => Box::new(AvgAccumulator::new(arg, input_schema)?),
        AggregateFunction::Min => Box::new(MinMaxAccumulator::new(arg, input_schema, MinOrMax::Min)?),
        AggregateFunction::Max => Box::new(MinMaxAccumulator::new(arg, input_schema, MinOrMax::Max)?),
    })
}

fn unwrap_aggregate(e: &Expr) -> EvalResult<(AggregateFunction, Expr)> {
    match e {
        Expr::Aggregate { fun, arg } => Ok((*fun, (**arg).clone())),
        Expr::Alias { expr, .. } => unwrap_aggregate(expr),
        other => Err(EvalError::NotImplemented(format!(
            "not an aggregate expression: {}",
            other.display_name()
        ))),
    }
}

// ------------------------------------------------------------------------
// COUNT(*)
// ------------------------------------------------------------------------

#[derive(Default)]
pub struct CountStarAccumulator {
    count: i64,
}

impl Accumulator for CountStarAccumulator {
    fn data_type(&self) -> DataType { DataType::Int64 }
    fn update_batch(&mut self, batch: &RecordBatch) -> EvalResult<()> {
        self.count += batch.num_rows() as i64;
        Ok(())
    }
    fn finalize(&mut self) -> EvalResult<ArrayRef> {
        Ok(Arc::new(Int64Array::from(vec![self.count])))
    }
}

// ------------------------------------------------------------------------
// COUNT(expr) — non-null
// ------------------------------------------------------------------------

pub struct CountAccumulator {
    arg: Expr,
    count: i64,
}

impl CountAccumulator {
    pub fn new(arg: Expr) -> Self { Self { arg, count: 0 } }
}

impl Accumulator for CountAccumulator {
    fn data_type(&self) -> DataType { DataType::Int64 }
    fn update_batch(&mut self, batch: &RecordBatch) -> EvalResult<()> {
        let arr = evaluate(&self.arg, batch)?;
        self.count += (arr.len() - arr.null_count()) as i64;
        Ok(())
    }
    fn finalize(&mut self) -> EvalResult<ArrayRef> {
        Ok(Arc::new(Int64Array::from(vec![self.count])))
    }
}

// ------------------------------------------------------------------------
// SUM with type promotion
// ------------------------------------------------------------------------

pub struct SumAccumulator {
    arg: Expr,
    target_dt: DataType,
    sum_i64: i64,
    sum_u64: u64,
    sum_f64: f64,
    seen_non_null: bool,
}

impl SumAccumulator {
    pub fn new(arg: Expr, input_schema: &Schema) -> EvalResult<Self> {
        let inner_dt = arg.data_type(input_schema).map_err(plume_err)?;
        let target_dt = sum_target_type(&inner_dt)?;
        Ok(Self {
            arg, target_dt,
            sum_i64: 0, sum_u64: 0, sum_f64: 0.0,
            seen_non_null: false,
        })
    }
}

fn sum_target_type(inner: &DataType) -> EvalResult<DataType> {
    Ok(match inner {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => DataType::Int64,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => DataType::UInt64,
        DataType::Float32 | DataType::Float64 => DataType::Float64,
        other => return Err(EvalError::NotImplemented(format!("SUM over {other:?}"))),
    })
}

impl Accumulator for SumAccumulator {
    fn data_type(&self) -> DataType { self.target_dt.clone() }

    fn update_batch(&mut self, batch: &RecordBatch) -> EvalResult<()> {
        let arr = evaluate(&self.arg, batch)?;
        let casted = cast(&arr, &self.target_dt)?;
        match &self.target_dt {
            DataType::Int64 => {
                if let Some(s) = aggregate::sum(casted.as_primitive::<Int64Type>()) {
                    self.sum_i64 = self.sum_i64.wrapping_add(s);
                    self.seen_non_null = true;
                }
            }
            DataType::UInt64 => {
                if let Some(s) = aggregate::sum(casted.as_primitive::<UInt64Type>()) {
                    self.sum_u64 = self.sum_u64.wrapping_add(s);
                    self.seen_non_null = true;
                }
            }
            DataType::Float64 => {
                if let Some(s) = aggregate::sum(casted.as_primitive::<Float64Type>()) {
                    self.sum_f64 += s;
                    self.seen_non_null = true;
                }
            }
            _ => unreachable!("sum_target_type produces only Int64/UInt64/Float64"),
        }
        Ok(())
    }

    fn finalize(&mut self) -> EvalResult<ArrayRef> {
        if !self.seen_non_null {
            return Ok(new_null_array(&self.target_dt, 1));
        }
        Ok(match &self.target_dt {
            DataType::Int64 => Arc::new(Int64Array::from(vec![self.sum_i64])),
            DataType::UInt64 => Arc::new(UInt64Array::from(vec![self.sum_u64])),
            DataType::Float64 => Arc::new(Float64Array::from(vec![self.sum_f64])),
            _ => unreachable!(),
        })
    }
}

// ------------------------------------------------------------------------
// AVG -> Float64
// ------------------------------------------------------------------------

pub struct AvgAccumulator {
    arg: Expr,
    sum: f64,
    count: i64,
}

impl AvgAccumulator {
    pub fn new(arg: Expr, input_schema: &Schema) -> EvalResult<Self> {
        let dt = arg.data_type(input_schema).map_err(plume_err)?;
        if !is_numeric(&dt) {
            return Err(EvalError::NotImplemented(format!("AVG over {dt:?}")));
        }
        Ok(Self { arg, sum: 0.0, count: 0 })
    }
}

impl Accumulator for AvgAccumulator {
    fn data_type(&self) -> DataType { DataType::Float64 }

    fn update_batch(&mut self, batch: &RecordBatch) -> EvalResult<()> {
        let arr = evaluate(&self.arg, batch)?;
        let casted = cast(&arr, &DataType::Float64)?;
        let typed = casted.as_primitive::<Float64Type>();
        if let Some(s) = aggregate::sum(typed) {
            self.sum += s;
        }
        self.count += (typed.len() - typed.null_count()) as i64;
        Ok(())
    }

    fn finalize(&mut self) -> EvalResult<ArrayRef> {
        if self.count == 0 {
            return Ok(new_null_array(&DataType::Float64, 1));
        }
        Ok(Arc::new(Float64Array::from(vec![self.sum / self.count as f64])))
    }
}

// ------------------------------------------------------------------------
// MIN / MAX
// ------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum MinOrMax { Min, Max }

pub struct MinMaxAccumulator {
    arg: Expr,
    target_dt: DataType,
    direction: MinOrMax,
    // Typed state — exactly one of these is active per `target_dt`.
    val_i64: Option<i64>,
    val_u64: Option<u64>,
    val_f64: Option<f64>,
    val_str: Option<String>,
}

impl MinMaxAccumulator {
    fn new(arg: Expr, input_schema: &Schema, direction: MinOrMax) -> EvalResult<Self> {
        let target_dt = arg.data_type(input_schema).map_err(plume_err)?;
        // Reject types we don't support yet rather than silently misbehaving.
        match &target_dt {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
            | DataType::Float32 | DataType::Float64
            | DataType::Utf8 | DataType::LargeUtf8 => {}
            other => {
                return Err(EvalError::NotImplemented(format!(
                    "MIN/MAX over {other:?}"
                )));
            }
        }
        Ok(Self {
            arg, target_dt, direction,
            val_i64: None, val_u64: None, val_f64: None, val_str: None,
        })
    }

    fn merge_i64(&mut self, candidate: i64) {
        self.val_i64 = Some(match self.val_i64 {
            None => candidate,
            Some(cur) => match self.direction {
                MinOrMax::Min => cur.min(candidate),
                MinOrMax::Max => cur.max(candidate),
            },
        });
    }
    fn merge_u64(&mut self, candidate: u64) {
        self.val_u64 = Some(match self.val_u64 {
            None => candidate,
            Some(cur) => match self.direction {
                MinOrMax::Min => cur.min(candidate),
                MinOrMax::Max => cur.max(candidate),
            },
        });
    }
    fn merge_f64(&mut self, candidate: f64) {
        // Use partial_cmp; treat NaN as larger than anything (we won't see it
        // from DuckDB-equivalent inputs — documented).
        self.val_f64 = Some(match self.val_f64 {
            None => candidate,
            Some(cur) => match self.direction {
                MinOrMax::Min => if candidate.partial_cmp(&cur).map(|o| o.is_lt()).unwrap_or(false) { candidate } else { cur },
                MinOrMax::Max => if candidate.partial_cmp(&cur).map(|o| o.is_gt()).unwrap_or(false) { candidate } else { cur },
            },
        });
    }
    fn merge_str(&mut self, candidate: &str) {
        self.val_str = Some(match self.val_str.take() {
            None => candidate.to_string(),
            Some(cur) => match self.direction {
                MinOrMax::Min => if candidate < cur.as_str() { candidate.to_string() } else { cur },
                MinOrMax::Max => if candidate > cur.as_str() { candidate.to_string() } else { cur },
            },
        });
    }
}

impl Accumulator for MinMaxAccumulator {
    fn data_type(&self) -> DataType { self.target_dt.clone() }

    fn update_batch(&mut self, batch: &RecordBatch) -> EvalResult<()> {
        let arr = evaluate(&self.arg, batch)?;
        match &self.target_dt {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                let casted = cast(&arr, &DataType::Int64)?;
                let typed = casted.as_primitive::<Int64Type>();
                if let Some(v) = match self.direction {
                    MinOrMax::Min => aggregate::min(typed),
                    MinOrMax::Max => aggregate::max(typed),
                } {
                    self.merge_i64(v);
                }
            }
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                let casted = cast(&arr, &DataType::UInt64)?;
                let typed = casted.as_primitive::<UInt64Type>();
                if let Some(v) = match self.direction {
                    MinOrMax::Min => aggregate::min(typed),
                    MinOrMax::Max => aggregate::max(typed),
                } {
                    self.merge_u64(v);
                }
            }
            DataType::Float32 | DataType::Float64 => {
                let casted = cast(&arr, &DataType::Float64)?;
                let typed = casted.as_primitive::<Float64Type>();
                if let Some(v) = match self.direction {
                    MinOrMax::Min => aggregate::min(typed),
                    MinOrMax::Max => aggregate::max(typed),
                } {
                    self.merge_f64(v);
                }
            }
            DataType::Utf8 => {
                let typed = arr.as_string::<i32>();
                let v = match self.direction {
                    MinOrMax::Min => aggregate::min_string(typed),
                    MinOrMax::Max => aggregate::max_string(typed),
                };
                if let Some(s) = v {
                    self.merge_str(s);
                }
            }
            DataType::LargeUtf8 => {
                let typed = arr.as_string::<i64>();
                let v = match self.direction {
                    MinOrMax::Min => aggregate::min_string(typed),
                    MinOrMax::Max => aggregate::max_string(typed),
                };
                if let Some(s) = v {
                    self.merge_str(s);
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn finalize(&mut self) -> EvalResult<ArrayRef> {
        Ok(match &self.target_dt {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                match self.val_i64 {
                    Some(v) => cast(
                        &(Arc::new(Int64Array::from(vec![v])) as ArrayRef),
                        &self.target_dt,
                    )?,
                    None => new_null_array(&self.target_dt, 1),
                }
            }
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                match self.val_u64 {
                    Some(v) => cast(
                        &(Arc::new(UInt64Array::from(vec![v])) as ArrayRef),
                        &self.target_dt,
                    )?,
                    None => new_null_array(&self.target_dt, 1),
                }
            }
            DataType::Float32 | DataType::Float64 => match self.val_f64 {
                Some(v) => cast(
                    &(Arc::new(Float64Array::from(vec![v])) as ArrayRef),
                    &self.target_dt,
                )?,
                None => new_null_array(&self.target_dt, 1),
            },
            DataType::Utf8 | DataType::LargeUtf8 => match self.val_str.take() {
                Some(v) => cast(
                    &(Arc::new(StringArray::from(vec![v])) as ArrayRef),
                    &self.target_dt,
                )?,
                None => new_null_array(&self.target_dt, 1),
            },
            _ => unreachable!(),
        })
    }
}

fn is_numeric(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
            | DataType::Float32 | DataType::Float64
    )
}

/// Convert plume_core::PlumeError into our EvalError so accumulators can use ?.
fn plume_err(e: plume_core::PlumeError) -> EvalError {
    EvalError::NotImplemented(format!("{e}"))
}

// ------------------------------------------------------------------------
// AggregateExec — global aggregation, no GROUP BY (Week 5).
// HashAggregateExec for grouping lands in Week 6.
// ------------------------------------------------------------------------

pub struct AggregateExec {
    input: Arc<dyn ExecutionPlan>,
    agg_exprs: Vec<Expr>,
    schema: SchemaRef,
}

impl AggregateExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        agg_exprs: Vec<Expr>,
        schema: SchemaRef,
    ) -> Self {
        Self { input, agg_exprs, schema }
    }
}

impl ExecutionPlan for AggregateExec {
    fn schema(&self) -> SchemaRef { self.schema.clone() }

    fn execute(&self) -> EvalResult<BatchStream> {
        let input_schema = self.input.schema();
        let mut accumulators: Vec<Box<dyn Accumulator>> = self
            .agg_exprs
            .iter()
            .map(|e| create_accumulator(e, input_schema.as_ref()))
            .collect::<EvalResult<_>>()?;

        // Drain the input stream synchronously. Global aggregation has to see
        // every row before it can emit anything, so there's no benefit to
        // staying lazy here.
        for batch in self.input.execute()? {
            let batch = batch?;
            for acc in &mut accumulators {
                acc.update_batch(&batch)?;
            }
        }

        let arrays = accumulators
            .iter_mut()
            .map(|a| a.finalize())
            .collect::<EvalResult<Vec<_>>>()?;
        let result = RecordBatch::try_new(self.schema.clone(), arrays)?;
        Ok(Box::new(std::iter::once(Ok(result))))
    }
}

// ------------------------------------------------------------------------
// HashAggregateExec — GROUP BY (Week 6).
//
// Row-at-a-time probe-and-update against a HashMap. Each row appends itself
// to its group's `Vec<Box<dyn Accumulator>>`. Output row order is the
// HashMap's iteration order — undefined, so all diff tests use the
// order-insensitive harness mode.
// ------------------------------------------------------------------------

/// Hashable, comparable representation of a group-key value. Float keys hash
/// by bit pattern, which means two NaNs with the same bit pattern collapse
/// into one group and otherwise form distinct ones. Documented; matches
/// what most engines do for f64 GROUP BY.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum KeyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(u64), // f64 -> bits
    Str(String),
}

fn extract_key(arr: &ArrayRef, row: usize) -> EvalResult<KeyValue> {
    if arr.is_null(row) {
        return Ok(KeyValue::Null);
    }
    Ok(match arr.data_type() {
        DataType::Boolean => KeyValue::Bool(arr.as_boolean().value(row)),
        DataType::Int8 => KeyValue::Int(arr.as_primitive::<arrow_array::types::Int8Type>().value(row) as i64),
        DataType::Int16 => KeyValue::Int(arr.as_primitive::<arrow_array::types::Int16Type>().value(row) as i64),
        DataType::Int32 => KeyValue::Int(arr.as_primitive::<Int32Type>().value(row) as i64),
        DataType::Int64 => KeyValue::Int(arr.as_primitive::<Int64Type>().value(row)),
        DataType::Float32 => KeyValue::Float((arr.as_primitive::<arrow_array::types::Float32Type>().value(row) as f64).to_bits()),
        DataType::Float64 => KeyValue::Float(arr.as_primitive::<Float64Type>().value(row).to_bits()),
        DataType::Utf8 => KeyValue::Str(arr.as_string::<i32>().value(row).to_string()),
        DataType::LargeUtf8 => KeyValue::Str(arr.as_string::<i64>().value(row).to_string()),
        other => return Err(EvalError::NotImplemented(format!("group key over {other:?}"))),
    })
}

/// Inverse of `extract_key` — assemble an `ArrayRef` of `target_dt` from the
/// per-group key values for one column. Used at finalize time.
fn build_group_column(target_dt: &DataType, values: &[KeyValue]) -> EvalResult<ArrayRef> {
    Ok(match target_dt {
        DataType::Boolean => {
            let mut b = BooleanBuilder::with_capacity(values.len());
            for v in values {
                match v {
                    KeyValue::Null => b.append_null(),
                    KeyValue::Bool(x) => b.append_value(*x),
                    other => return Err(EvalError::NotImplemented(format!("group key {other:?} into Boolean"))),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Int32 => {
            let mut b = Int32Builder::with_capacity(values.len());
            for v in values {
                match v {
                    KeyValue::Null => b.append_null(),
                    KeyValue::Int(x) => b.append_value(*x as i32),
                    other => return Err(EvalError::NotImplemented(format!("group key {other:?} into Int32"))),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Int64 => {
            let mut b = Int64Builder::with_capacity(values.len());
            for v in values {
                match v {
                    KeyValue::Null => b.append_null(),
                    KeyValue::Int(x) => b.append_value(*x),
                    other => return Err(EvalError::NotImplemented(format!("group key {other:?} into Int64"))),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Float64 => {
            let mut b = Float64Builder::with_capacity(values.len());
            for v in values {
                match v {
                    KeyValue::Null => b.append_null(),
                    KeyValue::Float(bits) => b.append_value(f64::from_bits(*bits)),
                    other => return Err(EvalError::NotImplemented(format!("group key {other:?} into Float64"))),
                }
            }
            Arc::new(b.finish())
        }
        DataType::Utf8 => {
            let mut b = StringBuilder::with_capacity(values.len(), values.len() * 8);
            for v in values {
                match v {
                    KeyValue::Null => b.append_null(),
                    KeyValue::Str(s) => b.append_value(s),
                    other => return Err(EvalError::NotImplemented(format!("group key {other:?} into Utf8"))),
                }
            }
            Arc::new(b.finish())
        }
        other => return Err(EvalError::NotImplemented(format!("group output column type {other:?}"))),
    })
}

pub struct HashAggregateExec {
    input: Arc<dyn ExecutionPlan>,
    group_exprs: Vec<Expr>,
    agg_exprs: Vec<Expr>,
    schema: SchemaRef,
}

impl HashAggregateExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        group_exprs: Vec<Expr>,
        agg_exprs: Vec<Expr>,
        schema: SchemaRef,
    ) -> Self {
        Self { input, group_exprs, agg_exprs, schema }
    }
}

impl ExecutionPlan for HashAggregateExec {
    fn schema(&self) -> SchemaRef { self.schema.clone() }

    fn execute(&self) -> EvalResult<BatchStream> {
        let input_schema = self.input.schema();
        let n_groups = self.group_exprs.len();
        let n_aggs = self.agg_exprs.len();

        // state: group_key -> per-aggregate accumulators
        let mut state: HashMap<Vec<KeyValue>, Vec<Box<dyn Accumulator>>> = HashMap::new();
        let agg_exprs = self.agg_exprs.clone();
        let group_exprs = self.group_exprs.clone();

        for batch in self.input.execute()? {
            let batch = batch?;
            // Pre-evaluate group_exprs once per batch.
            let group_arrays: Vec<ArrayRef> = group_exprs
                .iter()
                .map(|e| evaluate(e, &batch))
                .collect::<EvalResult<_>>()?;

            for row in 0..batch.num_rows() {
                let mut key = Vec::with_capacity(n_groups);
                for arr in &group_arrays {
                    key.push(extract_key(arr, row)?);
                }
                let row_batch = batch.slice(row, 1); // O(1) — adjusts offset only
                let accs = state.entry(key).or_insert_with(|| {
                    // create_accumulator's failure modes (unsupported types,
                    // missing columns) were all caught at plan-build time by
                    // LogicalPlan::aggregate's schema validation, so this
                    // expect is genuinely unreachable.
                    agg_exprs
                        .iter()
                        .map(|e| {
                            create_accumulator(e, input_schema.as_ref())
                                .expect("plan-time validation should have caught this")
                        })
                        .collect()
                });
                for acc in accs.iter_mut() {
                    acc.update_batch(&row_batch)?;
                }
            }
        }

        // Finalize. Drain state into parallel vecs so we can build group cols
        // (immutable key access) and agg cols (mutable accumulator finalize)
        // independently.
        let mut keys_per_group: Vec<Vec<KeyValue>> = Vec::with_capacity(state.len());
        let mut accs_per_group: Vec<Vec<Box<dyn Accumulator>>> = Vec::with_capacity(state.len());
        for (k, accs) in state.into_iter() {
            keys_per_group.push(k);
            accs_per_group.push(accs);
        }

        // Build group output columns (one per group_expr).
        let group_cols: Vec<ArrayRef> = (0..n_groups)
            .map(|col| {
                let dt = self.schema.field(col).data_type();
                let values: Vec<KeyValue> =
                    keys_per_group.iter().map(|k| k[col].clone()).collect();
                build_group_column(dt, &values)
            })
            .collect::<EvalResult<_>>()?;

        // Build aggregate output columns: finalize per group, concat across groups.
        let mut agg_cols: Vec<ArrayRef> = Vec::with_capacity(n_aggs);
        for agg_idx in 0..n_aggs {
            let one_row_arrays: Vec<ArrayRef> = accs_per_group
                .iter_mut()
                .map(|accs| accs[agg_idx].finalize())
                .collect::<EvalResult<_>>()?;
            let refs: Vec<&dyn Array> = one_row_arrays.iter().map(|a| a.as_ref()).collect();
            agg_cols.push(concat(&refs)?);
        }

        let mut all_cols = group_cols;
        all_cols.extend(agg_cols);
        let result = RecordBatch::try_new(self.schema.clone(), all_cols)?;
        Ok(Box::new(std::iter::once(Ok(result))))
    }
}

//! ORDER BY: a full sort (`SortExec`) and a bounded top-k (`TopKExec`).
//!
//! NULL placement follows each key's `nulls_first` flag (the logical
//! layer defaults it to NULLS LAST). Floats sort with Arrow's total order,
//! so NaN is greater than every number, as in DuckDB.

use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow::compute::{concat_batches, lexsort_to_indices, take_record_batch, SortColumn};
use arrow::row::{RowConverter, SortField};
use arrow_array::RecordBatch;
use arrow_schema::{SchemaRef, SortOptions};
use arrow_select::interleave::interleave_record_batch;
use plume_core::SortExpr;

use crate::compiled::{compile, CompiledExpr};
use crate::eval::EvalResult;
use crate::exec::{BatchStream, ExecutionPlan};
use crate::stat_value::StatValue;
use crate::topk_pruning::TopKBoundary;

/// A sort key compiled against the input schema.
#[derive(Clone)]
struct Key {
    expr: CompiledExpr,
    options: SortOptions,
}

fn compile_keys(exprs: &[SortExpr], input: &SchemaRef) -> Vec<Key> {
    exprs
        .iter()
        .map(|s| Key {
            expr: compile(&s.expr, input.as_ref()).expect(
                "sort key failed to compile against input schema; \
                 LogicalPlan::sort should have caught this at plan time",
            ),
            options: SortOptions {
                descending: !s.asc,
                nulls_first: s.nulls_first,
            },
        })
        .collect()
}

// ---- full sort ---------------------------------------------------------

pub struct SortExec {
    input: Arc<dyn ExecutionPlan>,
    keys: Vec<Key>,
    schema: SchemaRef,
}

impl SortExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, exprs: &[SortExpr]) -> Self {
        let schema = input.schema();
        let keys = compile_keys(exprs, &schema);
        Self {
            input,
            keys,
            schema,
        }
    }
}

impl ExecutionPlan for SortExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Drains the input, concatenates it, and emits one sorted batch (or
    /// nothing for empty input).
    fn execute(&self) -> EvalResult<BatchStream> {
        let batches = self.input.execute()?.collect::<EvalResult<Vec<_>>>()?;
        let all = concat_batches(&self.schema, &batches)?;
        if all.num_rows() == 0 {
            return Ok(Box::new(std::iter::empty()));
        }
        let columns = self
            .keys
            .iter()
            .map(|k| {
                Ok(SortColumn {
                    values: k.expr.evaluate(&all)?,
                    options: Some(k.options),
                })
            })
            .collect::<EvalResult<Vec<_>>>()?;
        let indices = lexsort_to_indices(&columns, None)?;
        let sorted = take_record_batch(&all, &indices)?;
        Ok(Box::new(std::iter::once(Ok(sorted))))
    }
}

// ---- top-k -------------------------------------------------------------

pub struct TopKExec {
    input: Arc<dyn ExecutionPlan>,
    keys: Vec<Key>,
    k: usize,
    schema: SchemaRef,
    /// Where to publish the first key of the current k-th best row, for
    /// a scan doing top-k pruning.
    boundary: Option<Arc<TopKBoundary>>,
}

impl TopKExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, exprs: &[SortExpr], k: usize) -> Self {
        let schema = input.schema();
        let keys = compile_keys(exprs, &schema);
        Self {
            input,
            keys,
            k,
            schema,
            boundary: None,
        }
    }

    /// Publish the k-th best first-key value to `boundary` whenever it
    /// changes. Once the heap is full, no row whose first key is strictly
    /// worse than this can enter it.
    pub fn with_boundary(mut self, boundary: Arc<TopKBoundary>) -> Self {
        self.boundary = Some(boundary);
        self
    }
}

/// One candidate row. `key` is the row-format encoding of its sort keys:
/// comparing the bytes compares the rows in ORDER BY order, so the
/// max-heap keeps the worst of the current best k on top.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Entry {
    key: Vec<u8>,
    batch: usize,
    row: usize,
}

/// The best k rows seen so far. Holds on to the batches those rows live
/// in and compacts them into one batch once they hold much more than k
/// rows, so memory stays O(k) rather than O(input).
struct TopKHeap {
    k: usize,
    heap: BinaryHeap<Entry>,
    batches: Vec<RecordBatch>,
    held_rows: usize,
}

impl TopKHeap {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.min(1 << 16)),
            batches: Vec::new(),
            held_rows: 0,
        }
    }

    /// Offer every row of `batch`; returns whether any was kept.
    fn insert(&mut self, batch: RecordBatch, keys: &arrow::row::Rows) -> EvalResult<bool> {
        let idx = self.batches.len();
        let mut used = false;
        for row in 0..batch.num_rows() {
            let key = keys.row(row);
            if self.heap.len() < self.k {
                self.heap.push(Entry {
                    key: key.as_ref().to_vec(),
                    batch: idx,
                    row,
                });
                used = true;
            } else if let Some(mut worst) = self.heap.peek_mut() {
                // Strictly better only: a row tied with the current k-th
                // can't improve the answer.
                if key.as_ref() < worst.key.as_slice() {
                    worst.key.clear();
                    worst.key.extend_from_slice(key.as_ref());
                    worst.batch = idx;
                    worst.row = row;
                    used = true;
                }
            }
        }
        if used {
            self.held_rows += batch.num_rows();
            self.batches.push(batch);
            if self.held_rows > 2 * self.k + 8192 {
                self.compact()?;
            }
        }
        Ok(used)
    }

    /// The current k-th best row as (batch, row), once the heap is full.
    fn kth(&self) -> Option<(&RecordBatch, usize)> {
        if self.heap.len() < self.k {
            return None;
        }
        let e = self.heap.peek()?;
        Some((&self.batches[e.batch], e.row))
    }

    /// Copy the rows still in the heap into a single batch.
    fn compact(&mut self) -> EvalResult<()> {
        let entries = std::mem::take(&mut self.heap).into_vec();
        let indices: Vec<(usize, usize)> = entries.iter().map(|e| (e.batch, e.row)).collect();
        let refs: Vec<&RecordBatch> = self.batches.iter().collect();
        let merged = interleave_record_batch(&refs, &indices)?;
        self.heap = entries
            .into_iter()
            .enumerate()
            .map(|(i, e)| Entry {
                key: e.key,
                batch: 0,
                row: i,
            })
            .collect();
        self.held_rows = merged.num_rows();
        self.batches = vec![merged];
        Ok(())
    }

    /// The kept rows, best first.
    fn finish(self) -> EvalResult<Option<RecordBatch>> {
        if self.heap.is_empty() {
            return Ok(None);
        }
        let indices: Vec<(usize, usize)> = self
            .heap
            .into_sorted_vec()
            .iter()
            .map(|e| (e.batch, e.row))
            .collect();
        let refs: Vec<&RecordBatch> = self.batches.iter().collect();
        Ok(Some(interleave_record_batch(&refs, &indices)?))
    }
}

impl ExecutionPlan for TopKExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Drains the input through a k-row heap and emits one batch of at
    /// most k rows in sort order.
    fn execute(&self) -> EvalResult<BatchStream> {
        if self.k == 0 {
            return Ok(Box::new(std::iter::empty()));
        }
        let fields = self
            .keys
            .iter()
            .map(|k| {
                Ok(SortField::new_with_options(
                    k.expr.data_type().clone(),
                    k.options,
                ))
            })
            .collect::<EvalResult<Vec<_>>>()?;
        let converter = RowConverter::new(fields)?;
        let mut heap = TopKHeap::new(self.k);
        for batch in self.input.execute()? {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            let cols = self
                .keys
                .iter()
                .map(|k| k.expr.evaluate(&batch))
                .collect::<EvalResult<Vec<_>>>()?;
            let rows = converter.convert_columns(&cols)?;
            let changed = heap.insert(batch, &rows)?;
            if let (true, Some(boundary)) = (changed, &self.boundary) {
                if let Some((batch, row)) = heap.kth() {
                    let first = self.keys[0].expr.evaluate(batch)?;
                    boundary.set(StatValue::from_array(&first, row));
                }
            }
        }
        Ok(Box::new(heap.finish()?.into_iter().map(Ok)))
    }
}

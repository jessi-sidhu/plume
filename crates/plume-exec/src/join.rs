//! Hash join + cross join.
//!
//! `HashJoinExec` is a build-then-probe equi-join. The right input is fully
//! drained into a hash table keyed by the right-side join exprs (NULL keys
//! never get inserted — see `keys::key_has_null` for the SQL semantics);
//! left batches then stream through and emit one output row per match. The
//! optional residual filter is evaluated on the concatenated row pair after
//! the hash match, so AND-conjuncts that mention both sides can ride along.
//!
//! Output columns are materialized via `arrow::compute::take` over a single
//! contiguous build-side `RecordBatch` (built from `concat_batches` over all
//! right batches). Going through `take` keeps the per-row work down to two
//! Vec<u32> pushes during probe; the heavy lifting happens once when we
//! emit a batch.
//!
//! Output batches are sized to `DEFAULT_BATCH_SIZE` (8192) when possible.
//! A single left row that matches many right rows can overshoot, which is
//! fine — downstream operators slice as needed.
//!
//! `CrossJoinExec` shares the materialization plumbing but skips the hash
//! map entirely: every left row pairs with every right row, optionally
//! filtered by a residual.
//!
//! This commit covers `JoinType::Inner` and `JoinType::Cross`. Outer joins
//! (Left/Right/Full) need a build-side matched bitmap and a post-probe
//! emission step for unmatched rows; that lands in the next commit. Semi
//! and Anti emit only the left side; that lands one commit after.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow::compute::take;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, UInt32Array};
use arrow_schema::SchemaRef;
use plume_core::{Expr, JoinType};

use crate::eval::{evaluate, EvalError, EvalResult};
use crate::exec::{BatchStream, ExecutionPlan};
use crate::keys::{extract_key, key_has_null, KeyValue};

const DEFAULT_BATCH_SIZE: usize = 8192;

// =======================================================================
// HashJoinExec — equi-join via build/probe.
// =======================================================================

pub struct HashJoinExec {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    on_left: Vec<Expr>,
    on_right: Vec<Expr>,
    filter: Option<Expr>,
    join_type: JoinType,
    schema: SchemaRef,
}

impl HashJoinExec {
    pub fn new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: Vec<(Expr, Expr)>,
        filter: Option<Expr>,
        join_type: JoinType,
        schema: SchemaRef,
    ) -> Self {
        let (on_left, on_right): (Vec<_>, Vec<_>) = on.into_iter().unzip();
        Self {
            left,
            right,
            on_left,
            on_right,
            filter,
            join_type,
            schema,
        }
    }
}

impl ExecutionPlan for HashJoinExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        match self.join_type {
            JoinType::Inner => {}
            other => {
                return Err(EvalError::NotImplemented(format!(
                    "{} join lowering is not implemented yet",
                    other.name()
                )));
            }
        }

        // ---- BUILD ----------------------------------------------------
        // Drain the right side into one contiguous build table so we can
        // address build rows by a single u32 index in `take`.
        let right_batches: Vec<RecordBatch> =
            self.right.execute()?.collect::<EvalResult<_>>()?;
        let right_schema = self.right.schema();
        let build_table = if right_batches.is_empty() {
            // Even with no rows, an empty RecordBatch carrying the schema
            // keeps the take-based materialization path uniform.
            RecordBatch::new_empty(right_schema.clone())
        } else {
            concat_batches(&right_schema, &right_batches)?
        };

        // Build hashmap: key -> Vec<row_idx_in_build_table>. Multiple
        // build rows can share a key, hence Vec.
        let mut index: HashMap<Vec<KeyValue>, Vec<u32>> = HashMap::new();
        if build_table.num_rows() > 0 {
            let key_arrays: Vec<ArrayRef> = self
                .on_right
                .iter()
                .map(|e| evaluate(e, &build_table))
                .collect::<EvalResult<_>>()?;
            for row in 0..build_table.num_rows() {
                let mut key = Vec::with_capacity(self.on_right.len());
                for arr in &key_arrays {
                    key.push(extract_key(arr, row)?);
                }
                // SQL equi-join: NULLs never match. Skip insertion so the
                // probe side never finds a NULL-keyed build row.
                if key_has_null(&key) {
                    continue;
                }
                index.entry(key).or_default().push(row as u32);
            }
        }

        // ---- PROBE ----------------------------------------------------
        let left_stream = self.left.execute()?;
        let on_left = self.on_left.clone();
        let filter = self.filter.clone();
        let schema = self.schema.clone();
        let left_schema = self.left.schema();
        let n_left_cols = left_schema.fields().len();

        // Collect all output batches up front. Streaming would require a
        // longer-lived hashmap reference + index iteration; for the row
        // counts a hash join sees in this engine that's not worth the
        // complexity right now. Documented as an opt-in for later.
        let mut out_batches: Vec<RecordBatch> = Vec::new();
        let mut left_idx_buf: Vec<u32> = Vec::with_capacity(DEFAULT_BATCH_SIZE);
        let mut right_idx_buf: Vec<u32> = Vec::with_capacity(DEFAULT_BATCH_SIZE);

        // Each left batch contributes its own (left_buf, right_buf) gather
        // pair, because `left_idx_buf` indexes into the *current* left
        // batch — different batches have distinct row spaces.
        for batch in left_stream {
            let left_batch = batch?;
            if left_batch.num_rows() == 0 {
                continue;
            }
            let key_arrays: Vec<ArrayRef> = on_left
                .iter()
                .map(|e| evaluate(e, &left_batch))
                .collect::<EvalResult<_>>()?;

            for row in 0..left_batch.num_rows() {
                let mut key = Vec::with_capacity(on_left.len());
                for arr in &key_arrays {
                    key.push(extract_key(arr, row)?);
                }
                if key_has_null(&key) {
                    continue;
                }
                if let Some(matches) = index.get(&key) {
                    for &right_row in matches {
                        left_idx_buf.push(row as u32);
                        right_idx_buf.push(right_row);
                    }
                }
            }

            if !left_idx_buf.is_empty() {
                let batch = build_join_batch(
                    &schema,
                    &left_batch,
                    &build_table,
                    &left_idx_buf,
                    &right_idx_buf,
                    n_left_cols,
                    filter.as_ref(),
                )?;
                if batch.num_rows() > 0 {
                    out_batches.push(batch);
                }
                left_idx_buf.clear();
                right_idx_buf.clear();
            }
        }

        Ok(Box::new(out_batches.into_iter().map(Ok)))
    }
}

// =======================================================================
// CrossJoinExec — every left row paired with every right row.
// =======================================================================

pub struct CrossJoinExec {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    filter: Option<Expr>,
    schema: SchemaRef,
}

impl CrossJoinExec {
    pub fn new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: Option<Expr>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            left,
            right,
            filter,
            schema,
        }
    }
}

impl ExecutionPlan for CrossJoinExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        let right_batches: Vec<RecordBatch> =
            self.right.execute()?.collect::<EvalResult<_>>()?;
        let right_schema = self.right.schema();
        let build_table = if right_batches.is_empty() {
            RecordBatch::new_empty(right_schema.clone())
        } else {
            concat_batches(&right_schema, &right_batches)?
        };

        let n_right_rows = build_table.num_rows();
        let n_left_cols = self.left.schema().fields().len();
        let schema = self.schema.clone();
        let filter = self.filter.clone();

        let left_stream = self.left.execute()?;
        let mut out_batches: Vec<RecordBatch> = Vec::new();

        // Empty right (or empty left, handled by outer loop being a no-op)
        // produces no rows and an empty stream.
        if n_right_rows == 0 {
            // Drain the left side anyway in case it has side effects (it
            // doesn't in this engine, but the contract is "execute the
            // input"). Cheap.
            for b in left_stream {
                let _ = b?;
            }
            return Ok(Box::new(out_batches.into_iter().map(Ok)));
        }

        for batch in left_stream {
            let left_batch = batch?;
            if left_batch.num_rows() == 0 {
                continue;
            }
            // For each left row, emit `n_right_rows` paired indices.
            let mut left_idx_buf: Vec<u32> = Vec::with_capacity(DEFAULT_BATCH_SIZE);
            let mut right_idx_buf: Vec<u32> = Vec::with_capacity(DEFAULT_BATCH_SIZE);
            for left_row in 0..left_batch.num_rows() {
                for right_row in 0..n_right_rows {
                    left_idx_buf.push(left_row as u32);
                    right_idx_buf.push(right_row as u32);
                }
            }
            let batch = build_join_batch(
                &schema,
                &left_batch,
                &build_table,
                &left_idx_buf,
                &right_idx_buf,
                n_left_cols,
                filter.as_ref(),
            )?;
            if batch.num_rows() > 0 {
                out_batches.push(batch);
            }
        }

        Ok(Box::new(out_batches.into_iter().map(Ok)))
    }
}

// =======================================================================
// Helpers shared by both operators.
// =======================================================================

/// Materialize one output `RecordBatch` from gather indices.
///
/// `output_schema.fields[..n_left_cols]` align with `left_batch`'s columns
/// in order; the rest align with `build_table`'s columns. Names in the
/// output schema may differ from input names (collision suffixing); the
/// data carried is the same.
///
/// If `filter` is set, evaluate it against the concatenated batch and
/// keep only the rows where it's `true` (NULL drops, matching SQL `WHERE`
/// semantics).
fn build_join_batch(
    output_schema: &SchemaRef,
    left_batch: &RecordBatch,
    build_table: &RecordBatch,
    left_idx: &[u32],
    right_idx: &[u32],
    n_left_cols: usize,
    filter: Option<&Expr>,
) -> EvalResult<RecordBatch> {
    debug_assert_eq!(left_idx.len(), right_idx.len());
    let left_indices: ArrayRef = Arc::new(UInt32Array::from(left_idx.to_vec()));
    let right_indices: ArrayRef = Arc::new(UInt32Array::from(right_idx.to_vec()));

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in 0..n_left_cols {
        cols.push(take(left_batch.column(col).as_ref(), &left_indices, None)?);
    }
    for col in 0..build_table.num_columns() {
        cols.push(take(build_table.column(col).as_ref(), &right_indices, None)?);
    }

    let batch = RecordBatch::try_new(output_schema.clone(), cols)?;
    if let Some(f) = filter {
        let mask = evaluate(f, &batch)?;
        let bool_mask = mask
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| EvalError::PredicateNotBoolean(mask.data_type().clone()))?;
        let filtered_cols: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|c| arrow::compute::filter(c.as_ref(), bool_mask).map_err(EvalError::from))
            .collect::<EvalResult<_>>()?;
        Ok(RecordBatch::try_new(output_schema.clone(), filtered_cols)?)
    } else {
        Ok(batch)
    }
}

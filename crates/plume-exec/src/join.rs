//! Hash join + cross join.
//!
//! `HashJoinExec` is a build-then-probe equi-join. The right input is fully
//! drained into a hash table keyed by the right-side join exprs (NULL keys
//! never get inserted — see `keys::key_has_null` for the SQL semantics);
//! left batches then stream through and emit output rows per the join type:
//!
//! - **Inner** — emit one row per (left, build) match. NULL-keyed left rows
//!   and left rows with no match are dropped.
//! - **Left** — same as Inner for matches. Left rows with no match
//!   (including NULL-keyed) are emitted with right columns NULL-padded.
//! - **Right** — same as Inner for matches, plus a per-build-row bitmap
//!   tracks which build rows ever matched. After the probe finishes,
//!   unmatched build rows are emitted with left columns NULL-padded.
//! - **Full** — Left + Right combined: emit unmatched left rows per probe
//!   batch and unmatched build rows after the loop.
//! - **Semi** — emit each left row that has *at least one* match, just the
//!   left columns, no duplicates per left row. Right is consulted for the
//!   existence check only.
//! - **Anti** — emit each left row with *no* match (including NULL-keyed),
//!   just the left columns.
//!
//! Optional residual filter is evaluated on the concatenated *matched* row
//! pair, so AND-conjuncts that mention both sides can ride along. The plan
//! layer rejects residuals on outer joins (folding a both-side predicate
//! into an outer-join residual would change semantics — that's the
//! optimizer's job), so the filter is reachable only for Inner / Cross.
//!
//! Output columns are materialized via `arrow::compute::take` over a single
//! contiguous build-side `RecordBatch` (built from `concat_batches` over all
//! right batches). Going through `take` keeps the per-row work down to two
//! Vec<u32> pushes during probe; the heavy lifting happens once when we
//! emit a batch. NULL-padded sides use `arrow_array::new_null_array` to
//! manufacture an all-null array of the right type and length — the join's
//! output schema already widens nullability for the side that may emit
//! NULLs (see `LogicalPlan::join`'s schema rules), so the resulting batch
//! validates against the schema cleanly.
//!
//! Output batches are sized to `DEFAULT_BATCH_SIZE` (8192) when possible.
//! A single left row that matches many right rows can overshoot, which is
//! fine — downstream operators slice as needed.
//!
//! `CrossJoinExec` shares the materialization plumbing but skips the hash
//! map entirely: every left row pairs with every right row, optionally
//! filtered by a residual.
//!
//! For Semi/Anti the output schema is exactly the left input's schema (no
//! right columns, no widening), as required by `LogicalPlan::join`'s rules.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow::compute::take;
use arrow_array::{new_null_array, Array, ArrayRef, BooleanArray, RecordBatch, UInt32Array};
use arrow_schema::{Schema, SchemaRef};
use plume_core::{Expr, JoinType};

use crate::eval::{evaluate, EvalError, EvalResult};
use crate::exec::{BatchStream, ExecutionPlan};
use crate::join_pruning::{key_range, JoinFilter, JoinKeyRange};
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
    /// (index into `on`, filter): publish the build keys' range for that
    /// key before the probe side executes. See `crate::join_pruning`.
    dynamic_filters: Vec<(usize, Arc<JoinFilter>)>,
    /// Send build keys as an exact set when there are at most this many
    /// distinct ones (0 = always min/max).
    key_set_max: usize,
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
            dynamic_filters: Vec::new(),
            key_set_max: 0,
        }
    }

    /// After building, publish the range of build key `on[i]` to each
    /// `(i, filter)`. Only correct when the scan reading the filter is on
    /// the probe (left) side of an Inner, Semi or Right join; the planner
    /// checks that.
    pub fn with_dynamic_filters(mut self, filters: Vec<(usize, Arc<JoinFilter>)>) -> Self {
        self.dynamic_filters = filters;
        self
    }

    /// Publish build keys as an exact set when there are at most `n`
    /// distinct ones (see `JoinKeyRange::Set`).
    pub fn with_key_set_max(mut self, n: usize) -> Self {
        self.key_set_max = n;
        self
    }
}

impl ExecutionPlan for HashJoinExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(&self) -> EvalResult<BatchStream> {
        match self.join_type {
            JoinType::Inner
            | JoinType::Left
            | JoinType::Right
            | JoinType::Full
            | JoinType::Semi
            | JoinType::Anti => {}
            JoinType::Cross => {
                return Err(EvalError::NotImplemented(
                    "cross join is handled by CrossJoinExec, not HashJoinExec".into(),
                ));
            }
        }
        let emit_match_pairs = matches!(
            self.join_type,
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
        );
        let emit_unmatched_left = matches!(self.join_type, JoinType::Left | JoinType::Full);
        let track_right_bitmap = matches!(self.join_type, JoinType::Right | JoinType::Full);
        let is_semi = matches!(self.join_type, JoinType::Semi);
        let is_anti = matches!(self.join_type, JoinType::Anti);
        let output_only_left = matches!(self.join_type, JoinType::Semi | JoinType::Anti);

        // ---- BUILD ----------------------------------------------------
        // Drain the right side into one contiguous build table so we can
        // address build rows by a single u32 index in `take`.
        let right_batches: Vec<RecordBatch> = self.right.execute()?.collect::<EvalResult<_>>()?;
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
        // Per-build-row matched bitmap (Right / Full only). NULL-keyed
        // build rows aren't in the index, so they're never set true and
        // are correctly emitted as unmatched-right rows for Right/Full.
        let mut matched_bitmap: Vec<bool> = if track_right_bitmap {
            vec![false; build_table.num_rows()]
        } else {
            Vec::new()
        };
        let key_arrays: Vec<ArrayRef> = if build_table.num_rows() > 0 {
            self.on_right
                .iter()
                .map(|e| evaluate(e, &build_table))
                .collect::<EvalResult<_>>()?
        } else {
            Vec::new()
        };
        // The build side is complete: tell the probe scan which keys can
        // possibly match, before it plans its reads.
        for (i, f) in &self.dynamic_filters {
            f.set(match key_arrays.get(*i) {
                Some(a) => key_range(a, self.key_set_max),
                None => JoinKeyRange::Empty,
            });
        }
        if build_table.num_rows() > 0 {
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
        let n_right_cols = build_table.num_columns();

        // Collect all output batches up front. Streaming would require a
        // longer-lived hashmap reference + index iteration; for the row
        // counts a hash join sees in this engine that's not worth the
        // complexity right now. Documented as an opt-in for later.
        let mut out_batches: Vec<RecordBatch> = Vec::new();
        let mut left_idx_buf: Vec<u32> = Vec::with_capacity(DEFAULT_BATCH_SIZE);
        let mut right_idx_buf: Vec<u32> = Vec::with_capacity(DEFAULT_BATCH_SIZE);
        let mut left_unmatched_buf: Vec<u32> = Vec::new();
        // Semi/Anti share this buffer — they're mutually exclusive, and
        // both emit a left-cols-only batch from it.
        let mut left_only_buf: Vec<u32> = Vec::new();

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
                // NULL-keyed left rows can never match any build row
                // (NULL ≠ NULL in equi-join). Where they go depends on
                // the join type: dropped for Inner/Right/Semi; emitted
                // as unmatched for Left/Full/Anti.
                if key_has_null(&key) {
                    if emit_unmatched_left {
                        left_unmatched_buf.push(row as u32);
                    }
                    if is_anti {
                        left_only_buf.push(row as u32);
                    }
                    continue;
                }
                match index.get(&key) {
                    Some(matches) => {
                        if emit_match_pairs {
                            for &right_row in matches {
                                left_idx_buf.push(row as u32);
                                right_idx_buf.push(right_row);
                                if track_right_bitmap {
                                    matched_bitmap[right_row as usize] = true;
                                }
                            }
                        }
                        if is_semi {
                            // Semi: one emission per matched left row,
                            // regardless of how many build rows matched.
                            left_only_buf.push(row as u32);
                        }
                    }
                    None => {
                        if emit_unmatched_left {
                            left_unmatched_buf.push(row as u32);
                        }
                        if is_anti {
                            left_only_buf.push(row as u32);
                        }
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
            if emit_unmatched_left && !left_unmatched_buf.is_empty() {
                let batch = build_left_unmatched_batch(
                    &schema,
                    &left_batch,
                    &left_unmatched_buf,
                    n_left_cols,
                    n_right_cols,
                    &build_table,
                )?;
                if batch.num_rows() > 0 {
                    out_batches.push(batch);
                }
                left_unmatched_buf.clear();
            }
            if output_only_left && !left_only_buf.is_empty() {
                let batch =
                    build_left_only_batch(&schema, &left_batch, &left_only_buf, n_left_cols)?;
                if batch.num_rows() > 0 {
                    out_batches.push(batch);
                }
                left_only_buf.clear();
            }
        }

        // ---- POST-PROBE: unmatched build rows (Right / Full) ----------
        if track_right_bitmap {
            let unmatched_right: Vec<u32> = matched_bitmap
                .iter()
                .enumerate()
                .filter_map(|(i, m)| if *m { None } else { Some(i as u32) })
                .collect();
            if !unmatched_right.is_empty() {
                let batch = build_right_unmatched_batch(
                    &schema,
                    &build_table,
                    &unmatched_right,
                    n_left_cols,
                    &left_schema,
                )?;
                if batch.num_rows() > 0 {
                    out_batches.push(batch);
                }
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
        let right_batches: Vec<RecordBatch> = self.right.execute()?.collect::<EvalResult<_>>()?;
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
        cols.push(take(
            build_table.column(col).as_ref(),
            &right_indices,
            None,
        )?);
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

/// Build an output batch where the *left* columns come from `left_batch` at
/// `left_idx`, and the *right* columns are entirely NULL of the right side's
/// data types. Used by Left and Full to emit unmatched left rows.
///
/// `build_table` is consulted only for its right-side data types — its row
/// data is not gathered. We use the build table rather than the live right
/// schema so the data types match the actual source arrays the matched-row
/// path also draws from (which `take` produces with their original types).
fn build_left_unmatched_batch(
    output_schema: &SchemaRef,
    left_batch: &RecordBatch,
    left_idx: &[u32],
    n_left_cols: usize,
    n_right_cols: usize,
    build_table: &RecordBatch,
) -> EvalResult<RecordBatch> {
    let left_indices: ArrayRef = Arc::new(UInt32Array::from(left_idx.to_vec()));
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(n_left_cols + n_right_cols);
    for col in 0..n_left_cols {
        cols.push(take(left_batch.column(col).as_ref(), &left_indices, None)?);
    }
    for col in 0..n_right_cols {
        cols.push(new_null_array(
            build_table.column(col).data_type(),
            left_idx.len(),
        ));
    }
    Ok(RecordBatch::try_new(output_schema.clone(), cols)?)
}

/// Gather only the left columns into a batch with `output_schema` (which
/// for Semi/Anti is exactly the left input's schema). Used by Semi (matched
/// left rows) and Anti (unmatched left rows).
fn build_left_only_batch(
    output_schema: &SchemaRef,
    left_batch: &RecordBatch,
    left_idx: &[u32],
    n_left_cols: usize,
) -> EvalResult<RecordBatch> {
    let left_indices: ArrayRef = Arc::new(UInt32Array::from(left_idx.to_vec()));
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(n_left_cols);
    for col in 0..n_left_cols {
        cols.push(take(left_batch.column(col).as_ref(), &left_indices, None)?);
    }
    Ok(RecordBatch::try_new(output_schema.clone(), cols)?)
}

/// Symmetric: right columns come from `build_table` at `right_idx`, left
/// columns are entirely NULL of the left input's data types. Used by Right
/// and Full to emit unmatched build rows after the probe completes.
fn build_right_unmatched_batch(
    output_schema: &SchemaRef,
    build_table: &RecordBatch,
    right_idx: &[u32],
    n_left_cols: usize,
    left_schema: &Schema,
) -> EvalResult<RecordBatch> {
    let right_indices: ArrayRef = Arc::new(UInt32Array::from(right_idx.to_vec()));
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(n_left_cols + build_table.num_columns());
    for col in 0..n_left_cols {
        cols.push(new_null_array(
            left_schema.field(col).data_type(),
            right_idx.len(),
        ));
    }
    for col in 0..build_table.num_columns() {
        cols.push(take(
            build_table.column(col).as_ref(),
            &right_indices,
            None,
        )?);
    }
    Ok(RecordBatch::try_new(output_schema.clone(), cols)?)
}

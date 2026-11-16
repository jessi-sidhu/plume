//! Scan metrics: how much of a Parquet file a scan actually touched.
//!
//! `ParquetScanExec` and `ParallelParquetScanExec` each own an
//! `Arc<ScanMetrics>`. Grab a handle with `.metrics()` before (or after)
//! handing the scan to the planner / an `Arc<dyn ExecutionPlan>`, run the
//! query, then read the counters.
//!
//! Counters describe the **most recent** `execute()` call: each call
//! resets them. Row-group and byte counts are known as soon as
//! `execute()` returns; `rows_decoded` grows as the stream is consumed.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parquet::file::metadata::ParquetMetaData;
use parquet::schema::types::SchemaDescriptor;

#[derive(Debug, Default)]
pub struct ScanMetrics {
    row_groups_total: AtomicU64,
    row_groups_pruned: AtomicU64,
    rows_decoded: AtomicU64,
    bytes_total: AtomicU64,
    bytes_read: AtomicU64,
    row_groups_full_match: AtomicU64,
    rows_filter_skipped: AtomicU64,
    row_groups_pruned_limit: AtomicU64,
    row_groups_read: AtomicU64,
    row_groups_pruned_topk: AtomicU64,
    /// Whether the batch the scan yielded most recently came from
    /// FullMatch row groups. Not a counter: it's how the scan tells a
    /// FilterExec directly above it that a batch needs no filtering. Only
    /// meaningful to that single consumer, which reads it right after
    /// pulling each batch.
    last_batch_full_match: AtomicBool,
}

/// Plain-value copy of a `ScanMetrics` at one moment.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ScanMetricsSnapshot {
    /// Row groups in the file.
    pub row_groups_total: u64,
    /// Row groups skipped by min/max pruning.
    pub row_groups_pruned: u64,
    /// Rows produced by the Parquet decoder (before any downstream Filter).
    pub rows_decoded: u64,
    /// Compressed size of the projected column chunks, all row groups.
    pub bytes_total: u64,
    /// Compressed size of the projected column chunks for the row groups
    /// the scan reads: the surviving ones, minus any a LIMIT hint or top-k
    /// pruning let it skip.
    pub bytes_read: u64,
    /// Surviving row groups whose stats prove every row matches the
    /// pushdown predicates.
    pub row_groups_full_match: u64,
    /// Rows a FilterExec passed through without evaluating its predicate,
    /// because they came from FullMatch row groups.
    pub rows_filter_skipped: u64,
    /// Surviving row groups the scan didn't plan to read because a LIMIT
    /// hint was already covered by FullMatch groups.
    pub row_groups_pruned_limit: u64,
    /// Row groups the reader actually decoded rows from. Lower than the
    /// surviving count when the consumer stopped early (e.g. a Limit).
    pub row_groups_read: u64,
    /// Surviving row groups skipped because a TopK above already held k
    /// rows strictly better than anything the group could contain.
    pub row_groups_pruned_topk: u64,
}

impl ScanMetrics {
    pub fn row_groups_total(&self) -> u64 {
        self.row_groups_total.load(Ordering::Relaxed)
    }
    pub fn row_groups_pruned(&self) -> u64 {
        self.row_groups_pruned.load(Ordering::Relaxed)
    }
    pub fn rows_decoded(&self) -> u64 {
        self.rows_decoded.load(Ordering::Relaxed)
    }
    pub fn bytes_total(&self) -> u64 {
        self.bytes_total.load(Ordering::Relaxed)
    }
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }
    pub fn row_groups_full_match(&self) -> u64 {
        self.row_groups_full_match.load(Ordering::Relaxed)
    }
    pub fn rows_filter_skipped(&self) -> u64 {
        self.rows_filter_skipped.load(Ordering::Relaxed)
    }
    pub fn row_groups_pruned_limit(&self) -> u64 {
        self.row_groups_pruned_limit.load(Ordering::Relaxed)
    }
    pub fn row_groups_read(&self) -> u64 {
        self.row_groups_read.load(Ordering::Relaxed)
    }
    pub fn row_groups_pruned_topk(&self) -> u64 {
        self.row_groups_pruned_topk.load(Ordering::Relaxed)
    }
    /// Top-k pruning skipped a surviving group of `bytes` projected bytes.
    pub(crate) fn add_row_group_pruned_topk(&self, bytes: u64) {
        self.row_groups_pruned_topk.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_sub(bytes, Ordering::Relaxed);
    }
    pub(crate) fn add_row_group_read(&self) {
        self.row_groups_read.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn set_last_batch_full_match(&self, full: bool) {
        self.last_batch_full_match.store(full, Ordering::Relaxed);
    }
    pub(crate) fn last_batch_full_match(&self) -> bool {
        self.last_batch_full_match.load(Ordering::Relaxed)
    }
    pub(crate) fn add_rows_filter_skipped(&self, n: usize) {
        self.rows_filter_skipped
            .fetch_add(n as u64, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ScanMetricsSnapshot {
        ScanMetricsSnapshot {
            row_groups_total: self.row_groups_total(),
            row_groups_pruned: self.row_groups_pruned(),
            rows_decoded: self.rows_decoded(),
            bytes_total: self.bytes_total(),
            bytes_read: self.bytes_read(),
            row_groups_full_match: self.row_groups_full_match(),
            rows_filter_skipped: self.rows_filter_skipped(),
            row_groups_pruned_limit: self.row_groups_pruned_limit(),
            row_groups_read: self.row_groups_read(),
            row_groups_pruned_topk: self.row_groups_pruned_topk(),
        }
    }

    /// Start of an `execute()` call: record the file-level numbers and
    /// zero `rows_decoded`.
    pub(crate) fn reset_for_execute(
        &self,
        metadata: &ParquetMetaData,
        schema: &SchemaDescriptor,
        projection: Option<&[String]>,
        surviving: &[usize],
        read: &[usize],
        full_match: usize,
    ) {
        let total = metadata.num_row_groups();
        let all: Vec<usize> = (0..total).collect();
        self.row_groups_total.store(total as u64, Ordering::Relaxed);
        self.row_groups_pruned
            .store((total - surviving.len()) as u64, Ordering::Relaxed);
        self.bytes_total.store(
            projected_chunk_bytes(metadata, schema, projection, &all),
            Ordering::Relaxed,
        );
        self.bytes_read.store(
            projected_chunk_bytes(metadata, schema, projection, read),
            Ordering::Relaxed,
        );
        self.rows_decoded.store(0, Ordering::Relaxed);
        self.row_groups_full_match
            .store(full_match as u64, Ordering::Relaxed);
        self.rows_filter_skipped.store(0, Ordering::Relaxed);
        self.row_groups_pruned_limit
            .store((surviving.len() - read.len()) as u64, Ordering::Relaxed);
        self.row_groups_read.store(0, Ordering::Relaxed);
        self.row_groups_pruned_topk.store(0, Ordering::Relaxed);
        self.last_batch_full_match.store(false, Ordering::Relaxed);
    }

    pub(crate) fn add_rows_decoded(&self, n: usize) {
        self.rows_decoded.fetch_add(n as u64, Ordering::Relaxed);
    }
}

/// Sum of compressed column-chunk sizes over `row_groups`, counting only
/// leaf columns whose top-level field is in `projection` (None = all).
pub(crate) fn projected_chunk_bytes(
    metadata: &ParquetMetaData,
    schema: &SchemaDescriptor,
    projection: Option<&[String]>,
    row_groups: &[usize],
) -> u64 {
    let root_fields = schema.root_schema().get_fields();
    let leaf_selected: Vec<bool> = (0..schema.num_columns())
        .map(|leaf| match projection {
            None => true,
            Some(cols) => {
                let root_name = root_fields[schema.get_column_root_idx(leaf)].name();
                cols.iter().any(|c| c == root_name)
            }
        })
        .collect();

    row_groups
        .iter()
        .map(|&rg_idx| {
            let rg = metadata.row_group(rg_idx);
            rg.columns()
                .iter()
                .zip(&leaf_selected)
                .filter(|(_, &keep)| keep)
                .map(|(chunk, _)| chunk.compressed_size().max(0) as u64)
                .sum::<u64>()
        })
        .sum()
}

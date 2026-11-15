//! Row-group pruning from Parquet min/max statistics.
//!
//! Shared by `ParquetScanExec` (sequential) and `ParallelParquetScanExec`
//! so both scans apply exactly the same rule.
//!
//! `row_group_can_satisfy` returns false only when we can prove the
//! predicate has no matches in this row group. When in doubt (missing
//! stats, type mismatches, NULL literal, exotic types) it returns true
//! so the row group is read and the per-row Filter handles correctness.

use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};
use parquet::file::statistics::Statistics;
use parquet::schema::types::Type;
use plume_core::{BinaryOperator, Expr, ScalarValue};

/// Indices of the row groups in `metadata` that might satisfy every
/// predicate in `predicates` (a conjunction). With no predicates, every
/// row group survives.
pub(crate) fn surviving_row_groups(
    metadata: &ParquetMetaData,
    parquet_root: &Type,
    predicates: &[Expr],
) -> Vec<usize> {
    (0..metadata.num_row_groups())
        .filter(|&rg_idx| {
            let rg = metadata.row_group(rg_idx);
            predicates
                .iter()
                .all(|pred| row_group_can_satisfy(pred, rg, parquet_root))
        })
        .collect()
}

pub(crate) fn row_group_can_satisfy(
    pred: &Expr,
    rg: &RowGroupMetaData,
    parquet_root: &Type,
) -> bool {
    // Pull `col OP lit` out of the predicate. The optimizer normalized to
    // column-on-the-left already, so we don't have to flip operators here.
    let Expr::BinaryOp { left, op, right } = pred else {
        return true;
    };
    let Expr::Column(name) = left.as_ref() else {
        return true;
    };
    let Expr::Literal(lit) = right.as_ref() else {
        return true;
    };

    let Some(col_idx) = parquet_root
        .get_fields()
        .iter()
        .position(|f| f.name() == name)
    else {
        return true; // unknown column — can't prune
    };
    let chunk = rg.column(col_idx);
    let Some(stats) = chunk.statistics() else {
        return true;
    };

    // Treat all-null row groups specially: any non-null comparison is false
    // for every row, so the group can be pruned.
    if let Some(null_count) = stats.null_count_opt() {
        if null_count == rg.num_rows() as u64 {
            return false;
        }
    }

    match (stats, lit) {
        (Statistics::Int32(s), ScalarValue::Int32(v)) => check_ord(
            s.min_opt().copied().map(i64::from),
            s.max_opt().copied().map(i64::from),
            *op,
            *v as i64,
        ),
        (Statistics::Int32(s), ScalarValue::Int64(v)) => check_ord(
            s.min_opt().copied().map(i64::from),
            s.max_opt().copied().map(i64::from),
            *op,
            *v,
        ),
        (Statistics::Int64(s), ScalarValue::Int32(v)) => {
            check_ord(s.min_opt().copied(), s.max_opt().copied(), *op, *v as i64)
        }
        (Statistics::Int64(s), ScalarValue::Int64(v)) => {
            check_ord(s.min_opt().copied(), s.max_opt().copied(), *op, *v)
        }
        (Statistics::Float(s), ScalarValue::Float64(v)) => check_ord_f64(
            s.min_opt().copied().map(f64::from),
            s.max_opt().copied().map(f64::from),
            *op,
            *v,
        ),
        (Statistics::Double(s), ScalarValue::Float64(v)) => {
            check_ord_f64(s.min_opt().copied(), s.max_opt().copied(), *op, *v)
        }
        (Statistics::ByteArray(s), ScalarValue::Utf8(v)) => {
            // ByteArray min/max are byte slices; interpret as UTF-8.
            let mn = s.min_opt().and_then(|b| std::str::from_utf8(b.data()).ok());
            let mx = s.max_opt().and_then(|b| std::str::from_utf8(b.data()).ok());
            check_ord_str(mn, mx, *op, v.as_str())
        }
        // Type mismatch / unsupported combo: assume the row group could match.
        _ => true,
    }
}

/// Generic ordered-type pruning check. Returns false only when proving the
/// predicate `col OP lit` cannot be true for any value in [min, max].
fn check_ord<T: Ord + Copy>(min: Option<T>, max: Option<T>, op: BinaryOperator, lit: T) -> bool {
    use BinaryOperator::*;
    let (Some(mn), Some(mx)) = (min, max) else {
        return true;
    };
    match op {
        Eq => !(lit < mn || lit > mx),
        // We can't exclude NotEq in general unless min == max == lit and
        // there are no nulls. Conservative: keep.
        NotEq => true,
        Lt => mn < lit,
        LtEq => mn <= lit,
        Gt => mx > lit,
        GtEq => mx >= lit,
        _ => true,
    }
}

/// f64 variant — uses partial_cmp; if any side is NaN the comparison is
/// unreliable so we don't prune.
fn check_ord_f64(min: Option<f64>, max: Option<f64>, op: BinaryOperator, lit: f64) -> bool {
    use BinaryOperator::*;
    let (Some(mn), Some(mx)) = (min, max) else {
        return true;
    };
    if mn.is_nan() || mx.is_nan() || lit.is_nan() {
        return true;
    }
    match op {
        Eq => !(lit < mn || lit > mx),
        NotEq => true,
        Lt => mn < lit,
        LtEq => mn <= lit,
        Gt => mx > lit,
        GtEq => mx >= lit,
        _ => true,
    }
}

fn check_ord_str(min: Option<&str>, max: Option<&str>, op: BinaryOperator, lit: &str) -> bool {
    use BinaryOperator::*;
    let (Some(mn), Some(mx)) = (min, max) else {
        return true;
    };
    match op {
        Eq => !(lit < mn || lit > mx),
        NotEq => true,
        Lt => mn < lit,
        LtEq => mn <= lit,
        Gt => mx > lit,
        GtEq => mx >= lit,
        _ => true,
    }
}

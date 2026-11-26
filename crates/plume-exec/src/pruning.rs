//! Row-group pruning from Parquet min/max statistics.
//!
//! Shared by `ParquetScanExec` (sequential) and `ParallelParquetScanExec`
//! so both scans apply exactly the same rule.
//!
//! Each row group is classified against the scan's pushdown predicates (a
//! conjunction of `col OP literal`):
//!
//! - `NoMatch`: the stats prove no row can satisfy the conjunction, so the
//!   group is skipped. This is exactly the old "prune" case.
//! - `FullMatch`: the stats prove every row satisfies every conjunct.
//!   That needs `null_count == 0` on each referenced column (a NULL row
//!   doesn't satisfy `col > 5`) and exact min/max values (string stats can
//!   be truncated).
//! - `PartialMatch`: anything else, including missing stats, unsupported
//!   types and NULL literals. When in doubt we say `PartialMatch`, never
//!   `FullMatch`.
//!
//! Floats: Plume's filter compares floats with Arrow's total order
//! (`-0.0 < 0.0`, NaN above everything), so `FullMatch` compares with
//! `f64::total_cmp` too. Parquet writers leave NaN out of min/max, so stats
//! can't show whether a group holds NaN. Like the existing `NoMatch` rule,
//! float classification assumes the data has no NaN.

use std::cmp::Ordering;

use parquet::basic::{ConvertedType, LogicalType, TimeUnit};
use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};
use parquet::file::statistics::Statistics;
use parquet::schema::types::Type;
use plume_core::{BinaryOperator, Expr, ScalarValue};

/// What a row group's statistics prove about a predicate.
// Named after the three states in the DataFusion LIMIT-pruning write-up.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowGroupMatch {
    /// No row can match: skip the group.
    NoMatch,
    /// Some rows might match: read the group and filter it.
    PartialMatch,
    /// Every row matches: read the group, no need to filter it.
    FullMatch,
}

/// Classify every row group in `metadata` against `predicates` (a
/// conjunction). With no predicates, every group is `FullMatch`.
pub(crate) fn classify_row_groups(
    metadata: &ParquetMetaData,
    parquet_root: &Type,
    predicates: &[Expr],
) -> Vec<RowGroupMatch> {
    (0..metadata.num_row_groups())
        .map(|rg_idx| classify_row_group(predicates, metadata.row_group(rg_idx), parquet_root))
        .collect()
}

/// Classify one row group against a conjunction: `NoMatch` if any
/// conjunct is `NoMatch`, `FullMatch` only if every conjunct is.
pub(crate) fn classify_row_group(
    predicates: &[Expr],
    rg: &RowGroupMetaData,
    parquet_root: &Type,
) -> RowGroupMatch {
    let mut all_full = true;
    for pred in predicates {
        match classify_predicate(pred, rg, parquet_root) {
            RowGroupMatch::NoMatch => return RowGroupMatch::NoMatch,
            RowGroupMatch::PartialMatch => all_full = false,
            RowGroupMatch::FullMatch => {}
        }
    }
    if all_full {
        RowGroupMatch::FullMatch
    } else {
        RowGroupMatch::PartialMatch
    }
}

/// Classify one row group against one `col OP literal` predicate.
pub(crate) fn classify_predicate(
    pred: &Expr,
    rg: &RowGroupMetaData,
    parquet_root: &Type,
) -> RowGroupMatch {
    use RowGroupMatch::*;
    if let Expr::IsNull(inner) | Expr::IsNotNull(inner) = pred {
        return classify_null_check(inner, matches!(pred, Expr::IsNull(_)), rg, parquet_root);
    }
    // Pull `col OP lit` out of the predicate. The optimizer normalized to
    // column-on-the-left already, so we don't have to flip operators here.
    let Expr::BinaryOp { left, op, right } = pred else {
        return PartialMatch;
    };
    let Expr::Column(name) = left.as_ref() else {
        return PartialMatch;
    };
    let Expr::Literal(lit) = right.as_ref() else {
        return PartialMatch;
    };

    let Some(col_idx) = parquet_root
        .get_fields()
        .iter()
        .position(|f| f.name() == name)
    else {
        return PartialMatch; // unknown column
    };
    let field = &parquet_root.get_fields()[col_idx];
    let chunk = rg.column(col_idx);
    let Some(stats) = chunk.statistics() else {
        return PartialMatch;
    };

    // All-null row group: any non-null comparison is false for every row.
    if let Some(null_count) = stats.null_count_opt() {
        if null_count == rg.num_rows() as u64 {
            return NoMatch;
        }
    }

    // FullMatch needs a known zero null count and exact min/max.
    let full_ok = stats.null_count_opt() == Some(0) && stats.min_is_exact() && stats.max_is_exact();
    let op = *op;

    match (stats, lit) {
        (Statistics::Int32(s), ScalarValue::Int32(v)) => classify_ord(
            s.min_opt().copied().map(i64::from),
            s.max_opt().copied().map(i64::from),
            op,
            *v as i64,
            full_ok,
        ),
        (Statistics::Int32(s), ScalarValue::Int64(v)) => classify_ord(
            s.min_opt().copied().map(i64::from),
            s.max_opt().copied().map(i64::from),
            op,
            *v,
            full_ok,
        ),
        (Statistics::Int64(s), ScalarValue::Int32(v)) => classify_ord(
            s.min_opt().copied(),
            s.max_opt().copied(),
            op,
            *v as i64,
            full_ok,
        ),
        (Statistics::Int64(s), ScalarValue::Int64(v)) => {
            classify_ord(s.min_opt().copied(), s.max_opt().copied(), op, *v, full_ok)
        }
        (Statistics::Float(s), ScalarValue::Float64(v)) => classify_f64(
            s.min_opt().copied().map(f64::from),
            s.max_opt().copied().map(f64::from),
            op,
            *v,
            full_ok,
        ),
        (Statistics::Double(s), ScalarValue::Float64(v)) => {
            classify_f64(s.min_opt().copied(), s.max_opt().copied(), op, *v, full_ok)
        }
        (Statistics::ByteArray(s), ScalarValue::Utf8(v)) => {
            // ByteArray min/max are byte slices; interpret as UTF-8.
            let mn = s.min_opt().and_then(|b| std::str::from_utf8(b.data()).ok());
            let mx = s.max_opt().and_then(|b| std::str::from_utf8(b.data()).ok());
            classify_ord(mn, mx, op, v.as_str(), full_ok)
        }
        // DATE is stored as physical INT32 (days since epoch) and
        // TIMESTAMP(MICROS) as physical INT64 (micros since epoch), so the
        // stats are plain integers. Only compare them once the column's
        // logical type confirms they mean the same thing as the literal;
        // otherwise (e.g. an INT32 that isn't a DATE, or MILLIS/NANOS
        // timestamps) don't classify.
        (Statistics::Int32(s), ScalarValue::Date32(v)) if is_date(field) => {
            classify_ord(s.min_opt().copied(), s.max_opt().copied(), op, *v, full_ok)
        }
        (Statistics::Int64(s), ScalarValue::TimestampMicros(v)) if is_timestamp_micros(field) => {
            classify_ord(s.min_opt().copied(), s.max_opt().copied(), op, *v, full_ok)
        }
        // Type mismatch / unsupported combo: the group could match.
        _ => PartialMatch,
    }
}

/// Classify `col IS NULL` (`is_null`) or `col IS NOT NULL` from the
/// group's null count. Only an exact count proves anything; without one
/// the group is `PartialMatch`.
fn classify_null_check(
    inner: &Expr,
    is_null: bool,
    rg: &RowGroupMetaData,
    parquet_root: &Type,
) -> RowGroupMatch {
    use RowGroupMatch::*;
    let Expr::Column(name) = inner else {
        return PartialMatch;
    };
    let Some(col_idx) = parquet_root
        .get_fields()
        .iter()
        .position(|f| f.name() == name)
    else {
        return PartialMatch;
    };
    let Some(nulls) = rg
        .column(col_idx)
        .statistics()
        .and_then(|s| s.null_count_opt())
    else {
        return PartialMatch;
    };
    let rows = rg.num_rows() as u64;
    let (none_null, all_null) = (nulls == 0, nulls == rows);
    match (is_null, none_null, all_null) {
        // An empty group is both; it has no rows to keep either way.
        (_, true, true) => NoMatch,
        (true, true, _) | (false, _, true) => NoMatch,
        (true, _, true) | (false, true, _) => FullMatch,
        _ => PartialMatch,
    }
}

/// Classify `col OP lit` for a group whose values lie in [min, max].
/// `full_ok` says whether the group may be called `FullMatch` at all (no
/// NULLs, exact stats).
fn classify_ord<T: Ord + Copy>(
    min: Option<T>,
    max: Option<T>,
    op: BinaryOperator,
    lit: T,
    full_ok: bool,
) -> RowGroupMatch {
    let (Some(mn), Some(mx)) = (min, max) else {
        return RowGroupMatch::PartialMatch;
    };
    classify_with(mn.cmp(&lit), mx.cmp(&lit), op, full_ok)
}

/// f64 variant. `NoMatch` keeps the IEEE comparisons the scan has always
/// used; `FullMatch` uses `total_cmp` to agree with Arrow's filter
/// kernels. Any NaN in the stats or the literal means `PartialMatch`.
fn classify_f64(
    min: Option<f64>,
    max: Option<f64>,
    op: BinaryOperator,
    lit: f64,
    full_ok: bool,
) -> RowGroupMatch {
    let (Some(mn), Some(mx)) = (min, max) else {
        return RowGroupMatch::PartialMatch;
    };
    if mn.is_nan() || mx.is_nan() || lit.is_nan() {
        return RowGroupMatch::PartialMatch;
    }
    if no_match_ieee(mn, mx, op, lit) {
        return RowGroupMatch::NoMatch;
    }
    match classify_with(mn.total_cmp(&lit), mx.total_cmp(&lit), op, full_ok) {
        RowGroupMatch::FullMatch => RowGroupMatch::FullMatch,
        _ => RowGroupMatch::PartialMatch,
    }
}

/// The scan's original float pruning test, unchanged.
fn no_match_ieee(mn: f64, mx: f64, op: BinaryOperator, lit: f64) -> bool {
    use BinaryOperator::*;
    match op {
        Eq => lit < mn || lit > mx,
        Lt => mn >= lit,
        LtEq => mn > lit,
        Gt => mx <= lit,
        GtEq => mx < lit,
        _ => false,
    }
}

/// Shared decision table, given how min and max compare to the literal.
fn classify_with(
    min_vs_lit: Ordering,
    max_vs_lit: Ordering,
    op: BinaryOperator,
    full_ok: bool,
) -> RowGroupMatch {
    use BinaryOperator::*;
    use Ordering::*;
    use RowGroupMatch::*;
    let (no_match, full) = match op {
        // No value in [min, max] equals lit / every value equals lit.
        Eq => (
            min_vs_lit == Greater || max_vs_lit == Less,
            min_vs_lit == Equal && max_vs_lit == Equal,
        ),
        // NotEq can't be proven empty from min/max (we'd need to know the
        // group is all `lit`), but it's full when lit is outside [min, max].
        NotEq => (false, min_vs_lit == Greater || max_vs_lit == Less),
        Lt => (min_vs_lit != Less, max_vs_lit == Less),
        LtEq => (min_vs_lit == Greater, max_vs_lit != Greater),
        Gt => (max_vs_lit != Greater, min_vs_lit == Greater),
        GtEq => (max_vs_lit == Less, min_vs_lit != Less),
        _ => (false, false),
    };
    if no_match {
        NoMatch
    } else if full && full_ok {
        FullMatch
    } else {
        PartialMatch
    }
}

fn is_date(field: &Type) -> bool {
    let info = field.get_basic_info();
    matches!(info.logical_type(), Some(LogicalType::Date))
        || info.converted_type() == ConvertedType::DATE
}

fn is_timestamp_micros(field: &Type) -> bool {
    let info = field.get_basic_info();
    match info.logical_type() {
        Some(LogicalType::Timestamp { unit, .. }) => matches!(unit, TimeUnit::MICROS(_)),
        Some(_) => false,
        None => info.converted_type() == ConvertedType::TIMESTAMP_MICROS,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parquet::basic::{LogicalType, Repetition, Type as PhysicalType};
    use parquet::data_type::ByteArray;
    use parquet::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};
    use parquet::file::statistics::{Statistics, ValueStatistics};
    use parquet::schema::types::{SchemaDescriptor, Type};
    use plume_core::{col, lit, lit_date, lit_timestamp, Expr};

    use super::{classify_predicate, classify_row_group, RowGroupMatch};
    use RowGroupMatch::*;

    const ROWS: i64 = 100;

    fn primitive(name: &str, physical: PhysicalType, logical: Option<LogicalType>) -> Type {
        Type::primitive_type_builder(name, physical)
            .with_repetition(Repetition::OPTIONAL)
            .with_logical_type(logical)
            .build()
            .unwrap()
    }

    /// A one-column schema plus a row group with `stats` for that column.
    fn group(field: Type, stats: Option<Statistics>) -> (Type, RowGroupMetaData) {
        let root = Type::group_type_builder("schema")
            .with_fields(vec![Arc::new(field)])
            .build()
            .unwrap();
        let descr = Arc::new(SchemaDescriptor::new(Arc::new(root.clone())));
        let mut chunk = ColumnChunkMetaData::builder(descr.column(0));
        if let Some(s) = stats {
            chunk = chunk.set_statistics(s);
        }
        let rg = RowGroupMetaData::builder(descr)
            .set_num_rows(ROWS)
            .set_column_metadata(vec![chunk.build().unwrap()])
            .build()
            .unwrap();
        (root, rg)
    }

    fn int64_group(min: i64, max: i64, nulls: Option<u64>) -> (Type, RowGroupMetaData) {
        group(
            primitive("x", PhysicalType::INT64, None),
            Some(Statistics::int64(Some(min), Some(max), None, nulls, false)),
        )
    }

    fn classify(pred: Expr, (root, rg): &(Type, RowGroupMetaData)) -> RowGroupMatch {
        classify_predicate(&pred, rg, root)
    }

    #[test]
    fn int_operators_and_boundaries() {
        // Values in [10, 20], no NULLs.
        let g = int64_group(10, 20, Some(0));
        let x = || col("x");
        let cases = [
            // =
            (x().eq(lit(5i64)), NoMatch),
            (x().eq(lit(10i64)), PartialMatch),
            (x().eq(lit(20i64)), PartialMatch),
            (x().eq(lit(21i64)), NoMatch),
            // <
            (x().lt(lit(10i64)), NoMatch), // min == lit
            (x().lt(lit(15i64)), PartialMatch),
            (x().lt(lit(20i64)), PartialMatch), // max == lit
            (x().lt(lit(21i64)), FullMatch),
            // <=
            (x().lt_eq(lit(9i64)), NoMatch),
            (x().lt_eq(lit(10i64)), PartialMatch), // min == lit
            (x().lt_eq(lit(19i64)), PartialMatch),
            (x().lt_eq(lit(20i64)), FullMatch), // max == lit
            // >
            (x().gt(lit(20i64)), NoMatch), // max == lit
            (x().gt(lit(15i64)), PartialMatch),
            (x().gt(lit(10i64)), PartialMatch), // min == lit
            (x().gt(lit(9i64)), FullMatch),
            // >=
            (x().gt_eq(lit(21i64)), NoMatch),
            (x().gt_eq(lit(20i64)), PartialMatch), // max == lit
            (x().gt_eq(lit(11i64)), PartialMatch),
            (x().gt_eq(lit(10i64)), FullMatch), // min == lit
            // != : full when lit is outside [min, max], never NoMatch
            (x().neq(lit(5i64)), FullMatch),
            (x().neq(lit(15i64)), PartialMatch),
        ];
        for (pred, want) in cases {
            let name = pred.display_name();
            assert_eq!(classify(pred, &g), want, "{name}");
        }
    }

    #[test]
    fn eq_is_full_only_when_min_max_and_lit_agree() {
        let g = int64_group(7, 7, Some(0));
        assert_eq!(classify(col("x").eq(lit(7i64)), &g), FullMatch);
        assert_eq!(classify(col("x").neq(lit(7i64)), &g), PartialMatch);
    }

    #[test]
    fn nulls_block_full_match() {
        // Same range, but 3 NULL rows: a NULL doesn't satisfy x > 5.
        let g = int64_group(10, 20, Some(3));
        assert_eq!(classify(col("x").gt(lit(5i64)), &g), PartialMatch);
        assert_eq!(classify(col("x").lt(lit(30i64)), &g), PartialMatch);
        // NoMatch is unaffected by NULLs.
        assert_eq!(classify(col("x").gt(lit(30i64)), &g), NoMatch);
    }

    #[test]
    fn unknown_null_count_blocks_full_match() {
        let g = int64_group(10, 20, None);
        assert_eq!(classify(col("x").gt(lit(5i64)), &g), PartialMatch);
    }

    #[test]
    fn all_null_group_is_no_match() {
        let g = int64_group(10, 20, Some(ROWS as u64));
        assert_eq!(classify(col("x").gt(lit(5i64)), &g), NoMatch);
    }

    #[test]
    fn missing_stats_or_unsupported_shapes_are_partial() {
        let g = group(primitive("x", PhysicalType::INT64, None), None);
        assert_eq!(classify(col("x").gt(lit(5i64)), &g), PartialMatch);

        let g = int64_group(10, 20, Some(0));
        // Column on both sides, unknown column, type mismatch.
        assert_eq!(classify(col("x").gt(col("x")), &g), PartialMatch);
        assert_eq!(classify(col("y").gt(lit(5i64)), &g), PartialMatch);
        assert_eq!(classify(col("x").gt(lit("a")), &g), PartialMatch);
    }

    #[test]
    fn int32_column() {
        let g = group(
            primitive("x", PhysicalType::INT32, None),
            Some(Statistics::int32(Some(-5), Some(5), None, Some(0), false)),
        );
        assert_eq!(classify(col("x").gt_eq(lit(-5)), &g), FullMatch);
        assert_eq!(classify(col("x").gt(lit(-5)), &g), PartialMatch);
        assert_eq!(classify(col("x").gt(lit(5)), &g), NoMatch);
        // Int64 literal against an INT32 column.
        assert_eq!(classify(col("x").lt(lit(6i64)), &g), FullMatch);
    }

    #[test]
    fn float_column() {
        let g = group(
            primitive("x", PhysicalType::DOUBLE, None),
            Some(Statistics::double(
                Some(1.5),
                Some(9.5),
                None,
                Some(0),
                false,
            )),
        );
        assert_eq!(classify(col("x").gt(lit(1.0)), &g), FullMatch);
        assert_eq!(classify(col("x").gt(lit(1.5)), &g), PartialMatch);
        assert_eq!(classify(col("x").lt_eq(lit(9.5)), &g), FullMatch);
        assert_eq!(classify(col("x").gt(lit(9.5)), &g), NoMatch);
        assert_eq!(classify(col("x").gt(lit(f64::NAN)), &g), PartialMatch);
    }

    #[test]
    fn float_signed_zero_follows_arrow_total_order() {
        // Arrow's filter treats -0.0 < 0.0, so a group whose min is -0.0
        // is not fully >= 0.0.
        let g = group(
            primitive("x", PhysicalType::DOUBLE, None),
            Some(Statistics::double(
                Some(-0.0),
                Some(3.0),
                None,
                Some(0),
                false,
            )),
        );
        assert_eq!(classify(col("x").gt_eq(lit(0.0)), &g), PartialMatch);
        assert_eq!(classify(col("x").gt_eq(lit(-0.0)), &g), FullMatch);
    }

    #[test]
    fn nan_in_stats_is_partial() {
        let g = group(
            primitive("x", PhysicalType::DOUBLE, None),
            Some(Statistics::double(
                Some(1.0),
                Some(f64::NAN),
                None,
                Some(0),
                false,
            )),
        );
        assert_eq!(classify(col("x").gt(lit(0.0)), &g), PartialMatch);
    }

    fn utf8_stats(min: &str, max: &str, max_exact: bool) -> Statistics {
        Statistics::ByteArray(
            ValueStatistics::new(
                Some(ByteArray::from(min)),
                Some(ByteArray::from(max)),
                None,
                Some(0),
                false,
            )
            .with_max_is_exact(max_exact),
        )
    }

    #[test]
    fn utf8_column() {
        let field = || primitive("s", PhysicalType::BYTE_ARRAY, Some(LogicalType::String));
        let g = group(field(), Some(utf8_stats("east", "west", true)));
        assert_eq!(classify(col("s").gt_eq(lit("east")), &g), FullMatch);
        assert_eq!(classify(col("s").lt(lit("west")), &g), PartialMatch);
        assert_eq!(classify(col("s").gt(lit("zzz")), &g), NoMatch);
        assert_eq!(classify(col("s").lt(lit("zzz")), &g), FullMatch);
    }

    #[test]
    fn truncated_string_stats_block_full_match() {
        // A truncated max is only a bound, so "every row < zzz" isn't proven.
        let field = primitive("s", PhysicalType::BYTE_ARRAY, Some(LogicalType::String));
        let g = group(field, Some(utf8_stats("east", "west", false)));
        assert_eq!(classify(col("s").lt(lit("zzz")), &g), PartialMatch);
    }

    #[test]
    fn date_column() {
        let field = || primitive("d", PhysicalType::INT32, Some(LogicalType::Date));
        // 1994-01-01 = 8766, 1994-12-31 = 9130 days since epoch.
        let g = group(
            field(),
            Some(Statistics::int32(
                Some(8766),
                Some(9130),
                None,
                Some(0),
                false,
            )),
        );
        let d = || col("d");
        assert_eq!(classify(d().gt_eq(lit_date("1994-01-01")), &g), FullMatch);
        assert_eq!(classify(d().gt(lit_date("1994-01-01")), &g), PartialMatch);
        assert_eq!(classify(d().lt(lit_date("1995-01-01")), &g), FullMatch);
        assert_eq!(classify(d().lt(lit_date("1994-01-01")), &g), NoMatch);

        // A plain INT32 (not DATE) column never classifies against a date.
        let plain = group(
            primitive("d", PhysicalType::INT32, None),
            Some(Statistics::int32(
                Some(8766),
                Some(9130),
                None,
                Some(0),
                false,
            )),
        );
        assert_eq!(
            classify(d().gt_eq(lit_date("1994-01-01")), &plain),
            PartialMatch
        );
    }

    #[test]
    fn timestamp_column() {
        let field = primitive(
            "t",
            PhysicalType::INT64,
            Some(LogicalType::Timestamp {
                is_adjusted_to_u_t_c: false,
                unit: parquet::basic::TimeUnit::MICROS(Default::default()),
            }),
        );
        // 2023-01-01 00:00:00 .. 2023-01-02 00:00:00 in micros.
        let lo = 1_672_531_200_000_000i64;
        let hi = lo + 86_400_000_000;
        let g = group(
            field,
            Some(Statistics::int64(Some(lo), Some(hi), None, Some(0), false)),
        );
        let t = || col("t");
        assert_eq!(
            classify(t().lt_eq(lit_timestamp("2023-01-02 00:00:00")), &g),
            FullMatch
        );
        assert_eq!(
            classify(t().lt(lit_timestamp("2023-01-02 00:00:00")), &g),
            PartialMatch
        );
        assert_eq!(
            classify(t().gt(lit_timestamp("2023-01-02 00:00:00")), &g),
            NoMatch
        );
    }

    #[test]
    fn conjunction_combines_conjuncts() {
        let (root, rg) = int64_group(10, 20, Some(0));
        let full = col("x").gt(lit(0i64));
        let partial = col("x").gt(lit(15i64));
        let none = col("x").gt(lit(99i64));
        let c = |preds: &[Expr]| classify_row_group(preds, &rg, &root);
        assert_eq!(c(&[]), FullMatch);
        assert_eq!(c(std::slice::from_ref(&full)), FullMatch);
        assert_eq!(c(&[full.clone(), partial.clone()]), PartialMatch);
        assert_eq!(c(&[full.clone(), none.clone()]), NoMatch);
        assert_eq!(c(&[partial, none]), NoMatch);
    }

    #[test]
    fn is_null_uses_exact_null_count() {
        let x = || col("x");
        // No NULLs: IS NULL skips, IS NOT NULL keeps everything.
        let g = int64_group(10, 20, Some(0));
        assert_eq!(classify(x().is_null(), &g), NoMatch);
        assert_eq!(classify(x().is_not_null(), &g), FullMatch);
        // Some NULLs: both partial.
        let g = int64_group(10, 20, Some(3));
        assert_eq!(classify(x().is_null(), &g), PartialMatch);
        assert_eq!(classify(x().is_not_null(), &g), PartialMatch);
        // All NULL: IS NOT NULL skips, IS NULL keeps everything.
        let g = int64_group(10, 20, Some(ROWS as u64));
        assert_eq!(classify(x().is_null(), &g), FullMatch);
        assert_eq!(classify(x().is_not_null(), &g), NoMatch);
    }

    #[test]
    fn null_check_without_a_count_is_partial() {
        let g = int64_group(10, 20, None);
        assert_eq!(classify(col("x").is_null(), &g), PartialMatch);
        assert_eq!(classify(col("x").is_not_null(), &g), PartialMatch);
        let g = group(primitive("x", PhysicalType::INT64, None), None);
        assert_eq!(classify(col("x").is_null(), &g), PartialMatch);
        // Unknown column, and a non-column operand.
        let g = int64_group(10, 20, Some(0));
        assert_eq!(classify(col("y").is_null(), &g), PartialMatch);
        assert_eq!(classify(col("x").gt(lit(5i64)).is_null(), &g), PartialMatch);
    }

    #[test]
    fn not_is_never_used_for_pruning() {
        // x > 99 alone would skip this group; NOT (x > 99) must not, and
        // NOT (x > 0) must not skip either, even though no row passes.
        let g = int64_group(10, 20, Some(0));
        assert_eq!(classify(not_(col("x").gt(lit(99i64))), &g), PartialMatch);
        assert_eq!(classify(not_(col("x").gt(lit(0i64))), &g), PartialMatch);
        assert_eq!(classify(not_(col("x").is_null()), &g), PartialMatch);
    }

    fn not_(e: Expr) -> Expr {
        plume_core::not(e)
    }
}

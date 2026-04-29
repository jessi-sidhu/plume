//! Criterion benchmarks for end-to-end pipelines on the 1M-row sales fixture.
//!
//! Measure four cuts that exercise the parts of the engine we've optimised:
//!
//!   1. project_unoptimized vs project_optimized
//!         -> column pruning impact (Week 8). The query reads two of five
//!            columns; the optimizer should drop the other three from the
//!            Parquet read.
//!
//!   2. filter_unoptimized vs filter_optimized
//!         -> column pruning + predicate pushdown / row-group skipping
//!            (Weeks 8-9). `price > 950` survives only the [875, 1000) row
//!            group — pruning takes 1M -> 125K rows of decode.
//!
//!   3. groupby_unoptimized vs groupby_optimized
//!         -> the GROUP BY hot path under HashAggregateExec (W6) plus the
//!            full optimizer pipeline (Weeks 8-10).
//!
//!   4. eval_treewalk vs eval_compiled
//!         -> compiled-expr (Week 11) vs the legacy tree-walking evaluator
//!            on the predicate `price > 500.0 AND region = 'west'`.
//!
//! Run with `cargo bench -p plume-bench`. Skip silently if the fixture
//! isn't present.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use plume_core::{avg, col, count_star, lit, sum, Expr};
use plume_exec::{collect, compile, evaluate};
use plume_io::scan_parquet_auto;
use plume_optimizer::Optimizer;

fn sales_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("fixtures/sales.parquet");
    if !p.exists() {
        eprintln!("SKIP: {} missing — run gen_fixtures first", p.display());
        return None;
    }
    Some(p)
}

fn build_project_plan(path: &std::path::Path) -> plume_core::LogicalPlan {
    scan_parquet_auto(path)
        .unwrap()
        .project(vec![col("order_id"), col("price")])
        .unwrap()
        .into_plan()
}

fn build_filter_plan(path: &std::path::Path) -> plume_core::LogicalPlan {
    scan_parquet_auto(path)
        .unwrap()
        .filter(col("price").gt(lit(950.0_f64)))
        .unwrap()
        .project(vec![col("order_id"), col("price")])
        .unwrap()
        .into_plan()
}

fn build_groupby_plan(path: &std::path::Path) -> plume_core::LogicalPlan {
    scan_parquet_auto(path)
        .unwrap()
        .filter(col("price").gt(lit(500.0_f64)))
        .unwrap()
        .aggregate(
            vec![col("region")],
            vec![count_star().alias("n"), avg(col("price")).alias("avg_price"), sum(col("quantity")).alias("total_qty")],
        )
        .unwrap()
        .into_plan()
}

fn bench_pipelines(c: &mut Criterion) {
    let Some(path) = sales_path() else { return };

    let mut group = c.benchmark_group("pipelines");
    group.sample_size(10);
    group.throughput(Throughput::Elements(1_000_000));

    let proj = build_project_plan(&path);
    let proj_opt = Optimizer::default().optimize(proj.clone()).unwrap();
    group.bench_function("project_unoptimized", |b| {
        b.iter(|| black_box(collect(&proj).unwrap()))
    });
    group.bench_function("project_optimized", |b| {
        b.iter(|| black_box(collect(&proj_opt).unwrap()))
    });

    let filt = build_filter_plan(&path);
    let filt_opt = Optimizer::default().optimize(filt.clone()).unwrap();
    group.bench_function("filter_unoptimized", |b| {
        b.iter(|| black_box(collect(&filt).unwrap()))
    });
    group.bench_function("filter_optimized", |b| {
        b.iter(|| black_box(collect(&filt_opt).unwrap()))
    });

    let gb = build_groupby_plan(&path);
    let gb_opt = Optimizer::default().optimize(gb.clone()).unwrap();
    group.bench_function("groupby_unoptimized", |b| {
        b.iter(|| black_box(collect(&gb).unwrap()))
    });
    group.bench_function("groupby_optimized", |b| {
        b.iter(|| black_box(collect(&gb_opt).unwrap()))
    });

    group.finish();
}

fn synthetic_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("price", DataType::Float64, false),
        Field::new("region", DataType::Utf8, false),
    ]));
    let prices: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
    let regions: Vec<&str> = (0..n)
        .map(|i| ["north", "south", "east", "west"][i % 4])
        .collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(prices)),
            Arc::new(StringArray::from(regions)),
        ],
    )
    .unwrap()
}

fn bench_evaluator(c: &mut Criterion) {
    // (price > 500.0) AND (region = 'west') over a synthetic 8192-row batch.
    let batch = synthetic_batch(8192);
    let predicate: Expr = col("price").gt(lit(500.0_f64)).and(col("region").eq(lit("west")));
    let compiled = compile(&predicate, batch.schema().as_ref()).unwrap();

    let mut group = c.benchmark_group("expression_evaluator");
    group.throughput(Throughput::Elements(8192));

    group.bench_function("treewalk", |b| {
        b.iter(|| {
            let arr = evaluate(&predicate, black_box(&batch)).unwrap();
            black_box(arr);
        })
    });
    group.bench_function("compiled", |b| {
        b.iter(|| {
            let arr = compiled.evaluate(black_box(&batch)).unwrap();
            black_box(arr);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_pipelines, bench_evaluator);
criterion_main!(benches);

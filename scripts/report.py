#!/usr/bin/env python3
"""Build docs/results-phase1.md and its charts from results/phase1/, and
docs/results-phase2.md from results/phase2/ (see report_phase2.py) when
that run exists.

Every number in the generated pages is read from the CSV and env.json
files. Nothing is typed by hand, so re-running this after a new
benchmark run updates the whole page.

    python3 scripts/report.py

Needs pandas and matplotlib.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase1"
DOCS = ROOT / "docs"
IMG = DOCS / "img"

LAYOUTS = ["random", "sorted", "compound"]
LAYOUT_COLORS = {"random": "#9aa0a6", "sorted": "#1a73e8", "compound": "#e8710a"}
DATASET_LABEL = {"tpch": "TPC-H SF1", "taxi": "NYC taxi"}
SWEEP_LABELS = {0.001: "0.1%", 0.01: "1%", 0.1: "10%", 0.5: "50%"}
SPEC_QUERIES = ["tpch_q1", "tpch_q6", "tpch_q12_join"]


def load():
    df = pd.read_csv(RESULTS / "pruning.csv")
    env = json.loads((RESULTS / "env.json").read_text())
    df["bytes_skipped_pct"] = 100.0 * (1 - df["bytes_read"] / df["bytes_total"])
    df["row_groups_skipped_pct"] = 100.0 * df["row_groups_pruned"] / df["row_groups_total"]
    df["layout"] = pd.Categorical(df["layout"], LAYOUTS, ordered=True)
    return df, env


def pct(x):
    """Percent with one decimal, or two significant digits below 0.1%."""
    if 0 < abs(x) < 0.1:
        return "{:.2g}%".format(x)
    return "{:.1f}%".format(x)


def save(fig, name, tight=True):
    IMG.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    # No Software/date metadata, so the PNGs are the same on every run.
    fig.savefig(IMG / name, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return "img/" + name


def chart_sweep(on):
    """Bytes skipped by layout x target selectivity, one panel per dataset."""
    sweep = on[on["target_selectivity"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, dataset in zip(axes, ["tpch", "taxi"]):
        d = sweep[sweep["dataset"] == dataset]
        targets = sorted(d["target_selectivity"].unique())
        width = 0.8 / len(LAYOUTS)
        for i, layout in enumerate(LAYOUTS):
            vals = [
                d[(d["layout"] == layout) & (d["target_selectivity"] == t)][
                    "bytes_skipped_pct"
                ].iloc[0]
                for t in targets
            ]
            xs = [j + (i - 1) * width for j in range(len(targets))]
            bars = ax.bar(xs, vals, width, label=layout, color=LAYOUT_COLORS[layout])
            # Label every bar so a 0% (random) bar is visibly 0, not missing.
            ax.bar_label(bars, labels=["{:.0f}".format(v) for v in vals], fontsize=7, padding=1)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([SWEEP_LABELS.get(t, str(t)) for t in targets])
        ax.set_xlabel("target selectivity (rows matched)")
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("bytes skipped by pruning (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="layout", loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return save(fig, "phase1-sweep-bytes-skipped.png", tight=False)


def chart_scatter(on):
    """Every query: actual selectivity vs bytes skipped, both datasets."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    markers = {"tpch": "o", "taxi": "^"}
    for dataset in ["tpch", "taxi"]:
        for layout in LAYOUTS:
            d = on[(on["dataset"] == dataset) & (on["layout"] == layout)]
            # Compound often lands exactly on sorted, so draw it hollow
            # and larger to keep both visible.
            hollow = layout == "compound"
            ax.scatter(
                d["actual_selectivity"] * 100,
                d["bytes_skipped_pct"],
                marker=markers[dataset],
                s=90 if hollow else 36,
                facecolors="none" if hollow else LAYOUT_COLORS[layout],
                edgecolors=LAYOUT_COLORS[layout],
                linewidths=1.5 if hollow else 0.5,
                label="{} / {}".format(DATASET_LABEL[dataset], layout),
            )
    ax.set_xscale("log")
    ax.set_xlabel("rows matched (% of file, log scale)")
    ax.set_ylabel("bytes skipped by pruning (%)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return save(fig, "phase1-selectivity-vs-skipped.png")


def chart_times(df):
    """Plume off / on vs DuckDB wall time per query, sorted layout."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dataset in zip(axes, ["tpch", "taxi"]):
        d = df[(df["dataset"] == dataset) & (df["layout"] == "sorted")]
        queries = list(dict.fromkeys(d["query"]))
        off = [d[(d["query"] == q) & (d["pruning"] == "off")]["plume_ms"].iloc[0] for q in queries]
        on_ = [d[(d["query"] == q) & (d["pruning"] == "on")]["plume_ms"].iloc[0] for q in queries]
        duck = [d[d["query"] == q]["duckdb_ms"].iloc[0] for q in queries]
        width = 0.27
        xs = range(len(queries))
        ax.bar([x - width for x in xs], off, width, label="Plume, pruning off", color="#9aa0a6")
        ax.bar(list(xs), on_, width, label="Plume, pruning on", color="#1a73e8")
        ax.bar([x + width for x in xs], duck, width, label="DuckDB", color="#188038")
        ax.set_yscale("log")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([q.split("_", 1)[1] for q in queries], rotation=35, ha="right", fontsize=8)
        ax.set_title("{} (sorted layout)".format(DATASET_LABEL[dataset]))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("median wall time, ms (log scale)")
    axes[0].legend(fontsize=8)
    return save(fig, "phase1-times.png")


def row(on, dataset, layout, query):
    m = on[(on["dataset"] == dataset) & (on["layout"] == layout) & (on["query"] == query)]
    if len(m) != 1:
        raise SystemExit("expected one row for {} {} {}".format(dataset, layout, query))
    return m.iloc[0]


def findings(df, on, env):
    out = []
    for dataset, q in [("tpch", "tpch_shipdate_1pct"), ("taxi", "taxi_pickup_1pct")]:
        s, r, c = (row(on, dataset, l, q) for l in ["sorted", "random", "compound"])
        out.append(
            "- **{}, ~1% of rows** (`{}`, actual {}): the sorted layout skipped {} of bytes "
            "({}/{} row groups), compound {}, random {}.".format(
                DATASET_LABEL[dataset], q, pct(100 * s["actual_selectivity"]),
                pct(s["bytes_skipped_pct"]), s["row_groups_pruned"], s["row_groups_total"],
                pct(c["bytes_skipped_pct"]), pct(r["bytes_skipped_pct"]),
            )
        )

    rnd = on[on["layout"] == "random"]
    out.append(
        "- **Random layout:** the most any query skipped was {} of bytes, across all {} "
        "random-layout measurements.".format(pct(rnd["bytes_skipped_pct"].max()), len(rnd))
    )

    spec = on[(on["layout"] == "sorted") & (on["query"].isin(SPEC_QUERIES))]
    out.append(
        "- **TPC-H spec queries** (Q1, Q6, Q12-style) match {}–{} of lineitem rows. On the "
        "sorted layout they skipped {}–{} of bytes.".format(
            pct(100 * spec["actual_selectivity"].min()), pct(100 * spec["actual_selectivity"].max()),
            pct(spec["bytes_skipped_pct"].min()), pct(spec["bytes_skipped_pct"].max()),
        )
    )

    q12 = row(on, "tpch", "sorted", "tpch_q12_join")
    out.append(
        "- **Correlated column:** Q12 filters on `l_receiptdate`, which no layout sorts on, "
        "but it tracks `l_shipdate`. On the shipdate-sorted file it still skipped {} of "
        "lineitem bytes.".format(pct(q12["bytes_skipped_pct"]))
    )

    fare = [row(on, "taxi", l, "taxi_fare_gt_100") for l in LAYOUTS]
    out.append(
        "- **Unsorted column:** `fare_amount > 100` (actual {} of rows) skipped {} / {} / {} "
        "of bytes on random / sorted / compound.".format(
            pct(100 * fare[0]["actual_selectivity"]),
            *[pct(f["bytes_skipped_pct"]) for f in fare]
        )
    )

    pu = [row(on, "taxi", l, "taxi_pulocation_eq_1") for l in LAYOUTS]
    out.append(
        "- **Point lookup:** `PULocationID = 1` (actual {} of rows) skipped {} / {} / {} of "
        "bytes on random / sorted / compound. Compound only sorts on `PULocationID` within "
        "equal pickup timestamps.".format(
            pct(100 * pu[0]["actual_selectivity"]), *[pct(p["bytes_skipped_pct"]) for p in pu]
        )
    )

    sweep_low = on[(on["layout"] == "sorted") & (on["target_selectivity"] == 0.001)]
    t = sweep_low[sweep_low["dataset"] == "tpch"].iloc[0]
    x = sweep_low[sweep_low["dataset"] == "taxi"].iloc[0]
    out.append(
        "- **Highly selective ranges** (~0.1% target) on the sorted layout skipped {} of bytes "
        "on TPC-H and {} on taxi.".format(pct(t["bytes_skipped_pct"]), pct(x["bytes_skipped_pct"]))
    )

    floor = (
        on.groupby("dataset")["row_groups_total"].first().rdiv(100.0)
    )
    out.append(
        "- **Row-group granularity:** one row group is {} of `lineitem` and {} of `trips`, so "
        "a query that touches any rows reads at least that much. Ranges narrower than one row "
        "group can't skip more.".format(pct(floor["tpch"]), pct(floor["taxi"]))
    )

    sorted_rows = df[df["layout"] == "sorted"]
    offs = sorted_rows[sorted_rows["pruning"] == "off"].set_index(["dataset", "query"])
    ons = sorted_rows[sorted_rows["pruning"] == "on"].set_index(["dataset", "query"])
    pruned = ons[ons["row_groups_pruned"] > 0].index
    speedup = (offs.loc[pruned, "plume_ms"] / ons.loc[pruned, "plume_ms"]).sort_values()
    best = speedup.index[-1]
    out.append(
        "- **Time:** on the sorted layout, for the {} queries where pruning skipped anything, "
        "Plume's median time dropped {:.1f}x to {:.1f}x (largest: `{}`, {:.0f} ms → {:.1f} ms).".format(
            len(speedup), speedup.iloc[0], speedup.iloc[-1], best[1],
            offs.loc[best, "plume_ms"], ons.loc[best, "plume_ms"],
        )
    )

    # Same work in both modes: nothing pruned, so off vs on is pure noise.
    rnd_all = df[df["layout"] == "random"]
    r_off = rnd_all[rnd_all["pruning"] == "off"].set_index(["dataset", "query"])["plume_ms"]
    r_on = rnd_all[rnd_all["pruning"] == "on"].set_index(["dataset", "query"])["plume_ms"]
    noise = (100.0 * (r_on - r_off).abs() / r_off).max()
    out.append(
        "- **Timing noise:** on the random layout nothing is pruned, so off and on do the same "
        "work. Their medians differ by up to {}, which is a rough measure of timing noise "
        "on this machine. Byte and row-group counts don't vary between runs.".format(pct(noise))
    )

    faster = int((on["duckdb_ms"] < on["plume_ms"]).sum())
    out.append(
        "- **DuckDB** ({} threads) was faster than Plume with pruning on (1 thread) in {} of "
        "{} measurements. It is here as a reference point, not a competitor.".format(
            env.get("duckdb_threads"), faster, len(on)
        )
    )
    return out


def table(on):
    lines = [
        "| dataset | query | target | actual | layout | row groups skipped | bytes skipped | rows decoded |",
        "|---|---|---|---|---|---|---|---|",
    ]
    d = on.sort_values(["dataset", "query", "layout"], ascending=[False, True, True])
    for _, r in d.iterrows():
        target = "" if pd.isna(r["target_selectivity"]) else pct(100 * r["target_selectivity"])
        lines.append(
            "| {} | `{}` | {} | {} | {} | {}/{} | {} | {:,} |".format(
                r["dataset"], r["query"], target, pct(100 * r["actual_selectivity"]),
                r["layout"], r["row_groups_pruned"], r["row_groups_total"],
                pct(r["bytes_skipped_pct"]), r["rows_decoded"],
            )
        )
    return lines


def time_table(df):
    lines = [
        "| dataset | query | layout | Plume off (ms) | Plume on (ms) | DuckDB (ms) |",
        "|---|---|---|---|---|---|",
    ]
    piv = df.pivot_table(
        index=["dataset", "query", "layout"], columns="pruning", values="plume_ms", observed=True
    )
    duck = df.groupby(["dataset", "query", "layout"], observed=True)["duckdb_ms"].first()
    for (dataset, query, layout) in sorted(piv.index, key=lambda k: (k[0] != "tpch", k[1], LAYOUTS.index(k[2]))):
        lines.append(
            "| {} | `{}` | {} | {:.1f} | {:.1f} | {:.1f} |".format(
                dataset, query, layout, piv.loc[(dataset, query, layout), "off"],
                piv.loc[(dataset, query, layout), "on"], duck[(dataset, query, layout)],
            )
        )
    return lines


def main():
    df, env = load()
    on = df[df["pruning"] == "on"].copy()

    sweep_img = chart_sweep(on)
    scatter_img = chart_scatter(on)
    times_img = chart_times(df)

    commit = (env.get("git_commit") or "unknown")[:7]
    md = [
        "# Phase 1 results: row-group pruning on real data",
        "",
        "<!-- Generated by scripts/report.py from results/phase1/. Don't edit by hand. -->",
        "",
        "How much of a Parquet file does min/max row-group pruning skip on real data, how much "
        "does the layout change that, and does TPC-H understate it the way the Snowflake paper "
        "reports ([Zimmerer et al. 2025](https://arxiv.org/abs/2504.11540), §8.3)?",
        "",
        "Data: TPC-H SF1 `lineitem` and NYC yellow taxi trips (Jan–Jun 2023), each written "
        "shuffled (`random`), sorted on the date column (`sorted`), and sorted on the date "
        "plus a second column (`compound`). Every file uses the same row-group size. See "
        "`crates/plume-bench/src/bin/prepare_datasets.rs` and "
        "`crates/plume-bench/src/workloads.rs`.",
        "",
        "Run: {} on {} ({} cores), {}, DuckDB {}, commit `{}`{}. Median of {} runs after "
        "{} warm-up. Plume is single-threaded; DuckDB used {} threads.".format(
            env.get("date_utc"), env.get("cpu"), env.get("cores"), env.get("rustc"),
            env.get("duckdb_version"), commit,
            " (uncommitted changes)" if env.get("git_dirty") else "",
            env.get("timed_runs"), env.get("warmup_runs"), env.get("duckdb_threads"),
        ),
        "",
        "Every Plume result was checked against DuckDB's, with pruning both on and off.",
        "",
        "## Findings",
        "",
    ]
    md += findings(df, on, env)
    md += [
        "",
        "## Bytes skipped by layout and selectivity",
        "",
        "![bytes skipped]({})".format(sweep_img),
        "",
        "## Selectivity vs. bytes skipped, every query",
        "",
        "![selectivity vs skipped]({})".format(scatter_img),
        "",
        "## Time",
        "",
        "![times]({})".format(times_img),
        "",
        "## All measurements (pruning on)",
        "",
        "Bytes are compressed column-chunk bytes for the columns the query reads, in the fact "
        "table (`lineitem` / `trips`). Rows decoded is what the Parquet reader produced "
        "before Plume's filter ran.",
        "",
    ]
    md += table(on)
    md += ["", "## Times", ""]
    md += time_table(df)
    md += [
        "",
        "## Reproduce",
        "",
        "```sh",
        "cargo run --release -p plume-bench --bin prepare_datasets",
        "cargo run --release -p plume-bench --bin prune_bench",
        "python3 scripts/report.py",
        "```",
        "",
    ]
    out = DOCS / "results-phase1.md"
    out.write_text("\n".join(md))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
    if (ROOT / "results" / "phase2" / "techniques.csv").exists():
        import report_phase2

        report_phase2.main()

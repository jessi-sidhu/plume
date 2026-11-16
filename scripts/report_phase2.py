"""Build docs/results-phase2.md and its charts from results/phase2/.

Run through scripts/report.py, which writes the Phase 1 page too. Every
number on the page is read from techniques.csv or env.json.

Each technique is measured with its own counter, in the run where that
technique is on:

- filter: row_groups_pruned (min/max stats vs the WHERE clause)
- limit:  row_groups_pruned_limit (groups a LIMIT hint let the scan skip)
- topk:   row_groups_pruned_topk (groups the TopK boundary ruled out)
- join:   row_groups_pruned_join (groups outside the build keys' range)

"% skipped" is that counter over the fact table's row groups.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase2"
DOCS = ROOT / "docs"
IMG = DOCS / "img"

TECHNIQUES = ["filter", "limit", "topk", "join"]
TECHNIQUE_LABEL = {
    "filter": "Filter (§3)",
    "limit": "LIMIT (§4)",
    "topk": "Top-k (§5)",
    "join": "Join (§6)",
}
COUNTER = {
    "filter": "row_groups_pruned",
    "limit": "row_groups_pruned_limit",
    "topk": "row_groups_pruned_topk",
    "join": "row_groups_pruned_join",
}
LAYOUTS = ["random", "sorted", "compound", "orderkey"]
LAYOUT_DESC = {
    "random": "shuffled",
    "sorted": "sorted on the date column",
    "compound": "sorted on the date, then a second column",
    "orderkey": "lineitem sorted on l_orderkey (TPC-H only)",
}
DATASETS = ["tpch", "taxi"]
DATASET_LABEL = {"tpch": "TPC-H SF1", "taxi": "NYC taxi"}
DATASET_COLORS = {"tpch": "#1a73e8", "taxi": "#e8710a"}
BINS = [0, 1e-9, 10, 25, 50, 75, 90, 100.0001]
BIN_LABELS = ["0", "(0,10]", "(10,25]", "(25,50]", "(50,75]", "(75,90]", "(90,100]"]


def pct(x):
    """Percent with one decimal, or two significant digits below 0.1%."""
    if 0 < abs(x) < 0.1:
        return "{:.2g}%".format(x)
    return "{:.1f}%".format(x)


def save(fig, name, tight=True):
    IMG.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(IMG / name, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return "img/" + name


def load():
    df = pd.read_csv(RESULTS / "techniques.csv")
    env = json.loads((RESULTS / "env.json").read_text())
    df["skipped"] = df.apply(lambda r: r[COUNTER[r["technique"]]], axis=1)
    df["skipped_pct"] = 100.0 * df["skipped"] / df["row_groups_total"]
    df["read_pct"] = 100.0 * df["row_groups_read"] / df["row_groups_total"]
    df["bytes_read_pct"] = 100.0 * df["bytes_read"] / df["bytes_total"]
    return df, env


def paired(df):
    """One row per (dataset, layout, query) with on/off side by side."""
    on = df[df["mode"] == "on"].set_index(["dataset", "layout", "query"])
    off = df[df["mode"] == "off"].set_index(["dataset", "layout", "query"])
    p = on.copy()
    for c in ["row_groups_read", "read_pct", "bytes_read", "bytes_read_pct", "rows_decoded", "plume_ms"]:
        p[c + "_off"] = off[c]
    return p.reset_index()


# ---- charts -------------------------------------------------------------


def chart_distribution(on):
    """Figure 1 style: share of measurements by % of row groups skipped,
    one panel per technique, TPC-H and taxi side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, t in zip(axes, TECHNIQUES):
        d = on[on["technique"] == t]
        width = 0.4
        for i, dataset in enumerate(DATASETS):
            dd = d[d["dataset"] == dataset]
            counts = pd.cut(dd["skipped_pct"], BINS, labels=BIN_LABELS, right=True,
                            include_lowest=True).value_counts().reindex(BIN_LABELS, fill_value=0)
            share = 100.0 * counts / max(len(dd), 1)
            xs = [j + (i - 0.5) * width for j in range(len(BIN_LABELS))]
            ax.bar(xs, share.values, width, color=DATASET_COLORS[dataset],
                   label="{} (n={})".format(DATASET_LABEL[dataset], len(dd)))
        ax.set_title(TECHNIQUE_LABEL[t])
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("% of row groups skipped", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("% of measurements")
    return save(fig, "phase2-distribution.png")


def chart_layouts(on):
    """Mean % skipped per technique and layout, one panel per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    colors = {"random": "#9aa0a6", "sorted": "#1a73e8", "compound": "#e8710a", "orderkey": "#188038"}
    for ax, dataset in zip(axes, DATASETS):
        d = on[on["dataset"] == dataset]
        layouts = [lay for lay in LAYOUTS if lay in set(d["layout"])]
        width = 0.8 / len(layouts)
        for i, layout in enumerate(layouts):
            vals = [d[(d["technique"] == t) & (d["layout"] == layout)]["skipped_pct"].mean()
                    for t in TECHNIQUES]
            xs = [j + (i - (len(layouts) - 1) / 2) * width for j in range(len(TECHNIQUES))]
            bars = ax.bar(xs, vals, width, label=layout, color=colors[layout])
            ax.bar_label(bars, labels=["{:.0f}".format(v) for v in vals], fontsize=7, padding=1)
        ax.set_xticks(range(len(TECHNIQUES)))
        ax.set_xticklabels([TECHNIQUE_LABEL[t] for t in TECHNIQUES], fontsize=8)
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="layout", fontsize=7)
    axes[0].set_ylabel("mean % of row groups skipped")
    return save(fig, "phase2-layouts.png")


# ---- tables -------------------------------------------------------------


def dataset_table(on):
    lines = [
        "| technique | dataset | measurements | skipped anything | median skipped | mean skipped | max skipped |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in TECHNIQUES:
        for dataset in DATASETS:
            d = on[(on["technique"] == t) & (on["dataset"] == dataset)]["skipped_pct"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                TECHNIQUE_LABEL[t], DATASET_LABEL[dataset], len(d), int((d > 0).sum()),
                pct(d.median()), pct(d.mean()), pct(d.max()),
            ))
    return lines


def layout_table(on):
    layouts = [lay for lay in LAYOUTS if lay in set(on["layout"])]
    lines = [
        "| technique | dataset | " + " | ".join(layouts) + " |",
        "|---|---|" + "---|" * len(layouts),
    ]
    for t in TECHNIQUES:
        for dataset in DATASETS:
            cells = []
            for layout in layouts:
                d = on[(on["technique"] == t) & (on["dataset"] == dataset) & (on["layout"] == layout)]
                cells.append("" if d.empty else pct(d["skipped_pct"].mean()))
            lines.append("| {} | {} | {} |".format(TECHNIQUE_LABEL[t], DATASET_LABEL[dataset], " | ".join(cells)))
    return lines


def detail_table(p):
    lines = [
        "| technique | dataset | query | layout | skipped by technique | row groups read (off → on) | rows decoded (off → on) | Plume ms (off → on) | DuckDB ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = {t: i for i, t in enumerate(TECHNIQUES)}
    p = p.assign(_t=p["technique"].map(order), _l=p["layout"].map({l: i for i, l in enumerate(LAYOUTS)}))
    p = p.sort_values(["_t", "dataset", "query", "_l"], ascending=[True, False, True, True])
    for _, r in p.iterrows():
        lines.append(
            "| {} | {} | `{}` | {} | {}/{} ({}) | {} → {} | {:,} → {:,} | {:.1f} → {:.1f} | {:.1f} |".format(
                r["technique"], r["dataset"], r["query"], r["layout"],
                int(r["skipped"]), int(r["row_groups_total"]), pct(r["skipped_pct"]),
                int(r["row_groups_read_off"]), int(r["row_groups_read"]),
                int(r["rows_decoded_off"]), int(r["rows_decoded"]),
                r["plume_ms_off"], r["plume_ms"], r["duckdb_ms"],
            )
        )
    return lines


# ---- findings -----------------------------------------------------------


def one(p, dataset, layout, query):
    m = p[(p["dataset"] == dataset) & (p["layout"] == layout) & (p["query"] == query)]
    if len(m) != 1:
        raise SystemExit("expected one row for {} {} {}".format(dataset, layout, query))
    return m.iloc[0]


def findings(on, p, env):
    out = []

    # §8.3: TPC-H vs taxi per technique, on the layouts both datasets have.
    shared = on[on["layout"].isin(["random", "sorted", "compound"])]
    for t in TECHNIQUES:
        d = shared[shared["technique"] == t]
        m = d.groupby("dataset")["skipped_pct"].mean()
        a = d.groupby("dataset")["skipped_pct"].apply(lambda s: int((s > 0).sum()))
        n = d.groupby("dataset").size()
        # Compare at the precision shown, so "10.2% vs 10.2%" reads as equal.
        mt, mx = round(m["tpch"], 1), round(m["taxi"], 1)
        if mt < mx:
            verdict = "TPC-H skipped less than taxi, as §8.3 predicts"
        elif mt > mx:
            verdict = "TPC-H skipped more than taxi, the opposite of what §8.3 predicts"
        else:
            verdict = "TPC-H and taxi skipped the same amount"
        out.append(
            "- **{}:** on the three layouts both datasets share, {}: mean {} of row groups on "
            "TPC-H ({} of {} measurements skipped anything) vs {} on taxi ({} of {}).".format(
                TECHNIQUE_LABEL[t], verdict, pct(m["tpch"]), a["tpch"], n["tpch"],
                pct(m["taxi"]), a["taxi"], n["taxi"],
            )
        )

    # LIMIT: groups read vs rows decoded, against a Limit that already
    # stops early.
    lim = p[p["technique"] == "limit"]
    fired = lim[lim["skipped"] > 0]
    less_rows = lim[lim["rows_decoded"] < lim["rows_decoded_off"]]
    more_rows = lim[lim["rows_decoded"] > lim["rows_decoded_off"]]
    ratio = (fired["rows_decoded_off"] / fired["rows_decoded"]).max() if len(fired) else 1.0
    out.append(
        "- **LIMIT pruning** planned to skip row groups in {} of {} LIMIT measurements, but "
        "row groups actually read went from {} (off) to {} (on) in total: the Limit operator "
        "stops the scan after 100 rows either way, so the groups it \"skipped\" would never "
        "have been reached. What it changes is which group is read first, a fully matching "
        "one instead of a partly matching one: rows decoded went down in {} measurements (up "
        "to {:.0f}x fewer), up in {}, and the same in the rest. Its skip counter overstates "
        "the saving for a single-threaded scan that stops early.".format(
            len(fired), len(lim), int(lim["row_groups_read_off"].sum()),
            int(lim["row_groups_read"].sum()), len(less_rows), ratio, len(more_rows),
        )
    )

    # Top-k: per query, which layouts let it prune.
    for dataset in DATASETS:
        for q in sorted(set(p[(p["technique"] == "topk") & (p["dataset"] == dataset)]["query"])):
            rows = p[(p["query"] == q) & (p["dataset"] == dataset)]
            parts = ", ".join(
                "{} {}/{}".format(r["layout"], int(r["skipped"]), int(r["row_groups_total"]))
                for _, r in rows.assign(_l=rows["layout"].map({l: i for i, l in enumerate(LAYOUTS)}))
                .sort_values("_l").iterrows()
            )
            out.append("- **Top-k `{}`:** row groups skipped per layout: {}.".format(q, parts))

    # Join: per query and layout.
    for dataset in DATASETS:
        for q in sorted(set(p[(p["technique"] == "join") & (p["dataset"] == dataset)]["query"])):
            rows = p[(p["query"] == q) & (p["dataset"] == dataset)]
            parts = ", ".join(
                "{} {}/{}".format(r["layout"], int(r["skipped"]), int(r["row_groups_total"]))
                for _, r in rows.assign(_l=rows["layout"].map({l: i for i, l in enumerate(LAYOUTS)}))
                .sort_values("_l").iterrows()
            )
            out.append("- **Join `{}`:** probe row groups skipped per layout: {}.".format(q, parts))

    # Speed where a technique fired.
    fired = p[(p["skipped"] > 0) & (p["technique"] != "filter")]
    if len(fired):
        sp = (fired["plume_ms_off"] / fired["plume_ms"]).sort_values()
        best = fired.loc[sp.index[-1]]
        out.append(
            "- **Time:** in the {} LIMIT/top-k/join measurements where the technique skipped "
            "something, Plume's median time changed by {:.2f}x to {:.1f}x (largest: `{}` on "
            "{} {}, {:.0f} ms → {:.1f} ms).".format(
                len(fired), sp.iloc[0], sp.iloc[-1], best["query"], best["dataset"],
                best["layout"], best["plume_ms_off"], best["plume_ms"],
            )
        )
    idle = p[(p["skipped"] == 0)]
    noise = (100.0 * (idle["plume_ms"] - idle["plume_ms_off"]).abs() / idle["plume_ms_off"]).max()
    out.append(
        "- **Timing noise:** where a technique skipped nothing, off and on do the same work; "
        "their medians differ by up to {}.".format(pct(noise))
    )
    return out


def main():
    df, env = load()
    on = df[df["mode"] == "on"].copy()
    p = paired(df)
    dist_img = chart_distribution(on)
    layout_img = chart_layouts(on)

    commit = (env.get("git_commit") or "unknown")[:7]
    md = [
        "# Phase 2 results: LIMIT, top-k and join pruning",
        "",
        "<!-- Generated by scripts/report.py from results/phase2/. Don't edit by hand. -->",
        "",
        "Phase 1 measured filter pruning. Phase 2 adds the paper's other three techniques "
        "([Zimmerer et al. 2025](https://arxiv.org/abs/2504.11540), §4–§6) and asks: how much "
        "does each skip on TPC-H vs real taxi data, across layouts, and does TPC-H give these "
        "techniques few opportunities, as the paper says (§8.3)?",
        "",
        "Layouts: " + "; ".join("`{}` = {}".format(k, v) for k, v in LAYOUT_DESC.items()) + ". "
        "Queries: `crates/plume-bench/src/workloads.rs` (`phase2_for_dataset`).",
        "",
        "Run: {} on {} ({} cores), {}, DuckDB {}, commit `{}`{}. Median of {} runs after {} "
        "warm-up. Plume is single-threaded; DuckDB used {} threads.".format(
            env.get("date_utc"), env.get("cpu"), env.get("cores"), env.get("rustc"),
            env.get("duckdb_version"), commit,
            " (uncommitted changes)" if env.get("git_dirty") else "",
            env.get("timed_runs"), env.get("warmup_runs"), env.get("duckdb_threads"),
        ),
        "",
        "Each query ran with its own technique off and on, every other technique left on, and "
        "both results were checked against DuckDB. LIMIT and top-k answers aren't unique, so "
        "those were checked by row count (or the sort columns, row by row) plus every row "
        "being in DuckDB's unlimited result. A technique's \"% skipped\" is its own counter "
        "over the fact table's row groups, so a group filter pruning already removed isn't "
        "counted again.",
        "",
        "## Findings",
        "",
    ]
    md += findings(on, p, env)
    md += [
        "",
        "## Distribution of row groups skipped, per technique",
        "",
        "In the style of the paper's Figure 1. One measurement is one query on one layout.",
        "",
        "![distribution]({})".format(dist_img),
        "",
        "## TPC-H vs taxi",
        "",
    ]
    md += dataset_table(on)
    md += [
        "",
        "## Which layout each technique depends on",
        "",
        "Mean % of row groups skipped.",
        "",
        "![layouts]({})".format(layout_img),
        "",
    ]
    md += layout_table(on)
    md += [
        "",
        "## All measurements",
        "",
        "\"Row groups read\" counts groups the Parquet reader actually decoded rows from, so it "
        "also reflects early termination (a Limit or TopK that stops pulling) and the other "
        "techniques, which stay on in both modes.",
        "",
    ]
    md += detail_table(p)
    md += [
        "",
        "## Reproduce",
        "",
        "```sh",
        "cargo run --release -p plume-bench --bin prepare_datasets",
        "cargo run --release -p plume-bench --bin prune_bench -- --phase 2",
        "python3 scripts/report.py",
        "```",
        "",
    ]
    out = DOCS / "results-phase2.md"
    out.write_text("\n".join(md))
    print("wrote", out.relative_to(ROOT))

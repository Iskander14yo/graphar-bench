#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gar_metrics
import numpy as np

LOADER_ORDER = ["gar", "neo4j-global", "neo4j-per-node", "pyg-inmem"]
RUN_TYPES = ["cold", "warm"]

_EXPS = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _mean(arr: list[float]) -> float:
    return float(np.mean(arr)) if arr else float("nan")


def _pct(arr: list[float], q: float) -> float:
    return float(np.percentile(arr, q)) if arr else float("nan")


def _peak(arr: list[float]) -> float:
    return float(max(arr)) if arr else float("nan")


def _fmt(v: float, spec: str = ".1f") -> str:
    return "n/a" if v != v else format(v, spec)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _is_run_dir(path: Path) -> bool:
    return (path / "run_info.json").exists()


def _find_latest_run(base: Path) -> Path:
    """Return the most recently modified run dir anywhere under base."""
    candidates = list(base.rglob("run_info.json"))
    if not candidates:
        sys.exit(f"No run directories found under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def _resolve_path(args: argparse.Namespace) -> Path:
    if args.results is None:
        path = _find_latest_run(_EXPS / "results")
        print(f"Auto-selected: {path}")
        return path
    path = args.results
    if not path.exists():
        sys.exit(f"Directory not found: {path}")
    return path


def load_results(path: Path) -> dict[str, dict[str, dict]]:
    """Load loader JSONs from a run dir or all run subdirs of a parent dir.

    Returns:
        {loader: {run_type: {"batches": [...], "system_metrics": [...], "epoch_times_ms": [...]}}}
    """
    run_dirs = [path] if _is_run_dir(path) else sorted(
        d for d in path.iterdir() if d.is_dir() and _is_run_dir(d)
    )
    if not run_dirs:
        sys.exit(f"No run directories found in {path}")

    agg: dict[str, dict[str, dict]] = {}
    for run_dir in run_dirs:
        for json_file in sorted(run_dir.glob("*.json")):
            if json_file.name == "run_info.json":
                continue
            loader = json_file.stem
            data = json.loads(json_file.read_text())
            agg.setdefault(loader, {})
            for run in data.get("runs", []):
                rt = run.get("type", "warm")
                bucket = agg[loader].setdefault(
                    rt, {
                        "batches": [],
                        "system_metrics": [],
                        "epoch_times_ms": [],
                        "feature_pipeline_timelines": [],
                    }
                )
                gar_metrics.init_bucket(bucket)
                bucket["batches"].extend(run.get("batches", []))
                bucket["system_metrics"].extend(run.get("system_metrics", []))
                if "feature_pipeline_timeline" in run:
                    bucket["feature_pipeline_timelines"].append({
                        "run_id": run.get("run_id"),
                        "samples": run["feature_pipeline_timeline"],
                    })
                gar_metrics.append_run(bucket, run)
                if "epoch_time_ms" in run:
                    bucket["epoch_times_ms"].append(run["epoch_time_ms"])
    return agg


def _non_profiled(batches: list[dict], loader: str) -> list[dict]:
    """Drop profiled batches for Neo4j (PROFILE adds real overhead)."""
    if loader.startswith("neo4j"):
        return [b for b in batches if not b.get("neo4j_profile")]
    return batches


def _ordered_loaders(agg: dict) -> list[str]:
    return [l for l in LOADER_ORDER if l in agg] + [
        l for l in sorted(agg) if l not in LOADER_ORDER
    ]


# ---------------------------------------------------------------------------
# Table data builders  (headers + rows, shared by ASCII and markdown output)
# ---------------------------------------------------------------------------

def _data_table1(agg: dict) -> tuple[list[str], list[list[str]]]:
    headers = ["Loader", "run", "mean (ms)", "P50 (ms)", "P95 (ms)", "Epoch (s)", "Batches/s", "Nodes", "Edges"]
    rows = []
    for loader in _ordered_loaders(agg):
        for rt in RUN_TYPES:
            if rt not in agg[loader]:
                continue
            bucket = agg[loader][rt]
            batches = _non_profiled(bucket["batches"], loader)
            totals = [b["total_ms"] for b in batches]
            epoch_times = bucket["epoch_times_ms"]
            # throughput: total batches / total epoch time avoids bias from unequal run counts
            if epoch_times:
                mean_epoch_s = _mean(epoch_times) / 1000
                batches_per_epoch = len(batches) / len(epoch_times)
                throughput = batches_per_epoch / mean_epoch_s
            else:
                mean_epoch_s = float("nan")
                throughput = float("nan")
            rows.append([
                loader, rt,
                _fmt(_mean(totals)),
                _fmt(_pct(totals, 50)),
                _fmt(_pct(totals, 95)),
                _fmt(mean_epoch_s),
                _fmt(throughput),
                _fmt(_mean([b["sampled_nodes"] for b in batches]), ".0f"),
                _fmt(_mean([b["sampled_edges"] for b in batches]), ".0f"),
            ])
    return headers, rows


def _data_table2(agg: dict) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Loader", "run",
        "Retr mean", "Retr P50", "Retr P95",
        "Conv mean", "Conv P50", "Conv P95",
        "Samp mean", "Samp P50", "Samp P95",
        "Feat mean", "Feat P50", "Feat P95",
    ]
    rows = []
    stage_loaders = [l for l in _ordered_loaders(agg) if l == "gar" or l.startswith("neo4j")]
    for loader in stage_loaders:
        is_gar = loader == "gar"
        for rt in RUN_TYPES:
            if rt not in agg[loader]:
                continue
            batches = _non_profiled(agg[loader][rt]["batches"], loader)
            retr = [b["retrieval_ms"] for b in batches]
            conv = [b["conversion_ms"] for b in batches]
            samp = [b["sampling_ms"] for b in batches if b.get("sampling_ms") is not None] if is_gar else []
            feat = [b["feature_fetch_ms"] for b in batches if b.get("feature_fetch_ms") is not None] if is_gar else []

            def _triple(arr: list[float]) -> list[str]:
                if arr:
                    return [_fmt(_mean(arr)), _fmt(_pct(arr, 50)), _fmt(_pct(arr, 95))]
                return ["n/a", "n/a", "n/a"]

            rows.append([loader, rt] + _triple(retr) + _triple(conv) + _triple(samp) + _triple(feat))
    return headers, rows


def _data_table3(agg: dict) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Loader", "run",
        "CPU mean (%)", "CPU peak (%)",
        "RAM mean (MB)", "RAM peak (MB)",
        "Disk mean (MB/s)", "Disk peak (MB/s)",
    ]
    rows = []
    for loader in _ordered_loaders(agg):
        for rt in RUN_TYPES:
            if rt not in agg[loader]:
                continue
            metrics = agg[loader][rt]["system_metrics"]
            if not metrics:
                rows.append([loader, rt] + ["n/a"] * 6)
                continue
            cpu  = [m["cpu_pct"] for m in metrics]
            rss  = [m["rss_mb"] for m in metrics]
            disk = [m["disk_read_mb_s"] for m in metrics]
            rows.append([
                loader, rt,
                _fmt(_mean(cpu)), _fmt(_peak(cpu)),
                _fmt(_mean(rss)), _fmt(_peak(rss)),
                _fmt(_mean(disk)), _fmt(_peak(disk)),
            ])
    return headers, rows


def _data_neo4j_profile(agg: dict) -> tuple[list[str], list[list[str]]]:
    profiled = [
        b
        for loader in agg if loader.startswith("neo4j")
        for rt_data in agg[loader].values()
        for b in rt_data["batches"] if b.get("neo4j_profile")
    ]
    if not profiled:
        return [], []
    db_hits     = [b["neo4j_profile"].get("db_hits", 0) for b in profiled]
    cache_hits  = [b["neo4j_profile"].get("page_cache_hits", 0) for b in profiled]
    cache_miss  = [b["neo4j_profile"].get("page_cache_misses", 0) for b in profiled]
    op_times    = [b["neo4j_profile"].get("total_op_time_ms", 0) for b in profiled]
    headers = ["Stat", "DB Hits", "Cache Hits", "Cache Misses", "Total Op Time (ms)"]
    rows = [[
        f"mean (N={len(profiled)})",
        _fmt(_mean(db_hits), ".0f"),
        _fmt(_mean(cache_hits), ".0f"),
        _fmt(_mean(cache_miss), ".0f"),
        _fmt(_mean(op_times)),
    ]]
    return headers, rows


def _data_chunk_manager(agg: dict) -> tuple[list[str], list[list[str]]]:
    return gar_metrics.data_chunk_manager(agg, _ordered_loaders(agg), RUN_TYPES, _fmt)


def _data_feature_cursor(agg: dict) -> tuple[list[str], list[list[str]]]:
    return gar_metrics.data_feature_cursor(
        agg,
        _ordered_loaders(agg),
        RUN_TYPES,
        _fmt,
        _non_profiled,
    )


def _data_feature_pipeline(agg: dict) -> tuple[list[str], list[list[str]]]:
    return gar_metrics.data_feature_pipeline(
        agg,
        _ordered_loaders(agg),
        RUN_TYPES,
        _fmt,
    )


# ---------------------------------------------------------------------------
# Save table images
# ---------------------------------------------------------------------------

_TABLES = [
    ("Table 1: Main comparison",                 _data_table1),
    ("Table 2: Stage breakdown (GAR and Neo4j)", _data_table2),
    ("Table 3: Resource usage",                  _data_table3),
    ("Chunk cache summary",                      _data_chunk_manager),
    ("Feature pipeline summary",                 _data_feature_pipeline),
    ("Feature cursor summary",                   _data_feature_cursor),
    ("Neo4j PROFILE summary",                    _data_neo4j_profile),
]



def _render_table(ax, title: str, headers: list[str], rows: list[list[str]]) -> None:
    n_cols = len(headers)
    n_rows = len(rows)
    ax.axis("off")

    char_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    total_chars = sum(char_widths)
    font_size = 9 if n_cols <= 8 else 7

    col_w = [cw / total_chars for cw in char_widths]
    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_w,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)

    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2C5F8A")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#1a3a5c")

    for i in range(1, n_rows + 1):
        bg = "#EBF3FB" if i % 2 == 0 else "#FFFFFF"
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#C8D8EA")

    ax.set_title(title, fontweight="bold", fontsize=font_size + 1, pad=10)


def save_table_images(agg: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    built = [(title, *builder(agg)) for title, builder in _TABLES]
    built = [(t, h, r) for t, h, r in built if r]
    if not built:
        return

    # Figure width scales with widest table; heights scale with each table's row count
    fig_w = 0.0
    heights = []
    for _, headers, rows in built:
        char_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        fig_w = max(fig_w, sum(char_widths) * 0.13)
        heights.append(max(1.2, len(rows) * 0.38 + 1.0))
    fig_w = max(6.0, min(fig_w, 26.0))
    fig_h = sum(heights)

    fig, axes = plt.subplots(
        len(built), 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": heights},
    )
    if len(built) == 1:
        axes = [axes]
    for ax, (title, headers, rows) in zip(axes, built):
        _render_table(ax, title, headers, rows)

    path = out_dir / "tables.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _best_run_type(agg: dict, loader: str) -> str:
    return "warm" if "warm" in agg[loader] else "cold"


def _stitched_time_seconds(metrics: list[dict], gap_s: float = 1.0) -> list[float]:
    """Monotonic wall-clock-like time from sample timestamps (handles merged runs)."""
    out: list[float] = []
    offset = 0.0
    prev_raw: float | None = None
    for m in metrics:
        raw = m["timestamp_ms"] / 1000.0
        if prev_raw is not None and raw < prev_raw - 0.5:
            offset = out[-1] + gap_s - raw
        out.append(offset + raw)
        prev_raw = raw
    return out


def _plot_system_resources_timeseries(agg: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for loader in _ordered_loaders(agg):
        for rt in RUN_TYPES:
            if rt not in agg[loader]:
                continue
            metrics = agg[loader][rt]["system_metrics"]
            if not metrics:
                continue
            ts = _stitched_time_seconds(metrics)
            cpu = [m["cpu_pct"] for m in metrics]
            rss = [m["rss_mb"] for m in metrics]
            disk = [m["disk_read_mb_s"] for m in metrics]

            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            fig.suptitle(f"System resources — {loader} / {rt}")

            axes[0].plot(ts, cpu, color="#2C5F8A", linewidth=1.0)
            axes[0].set_ylabel("CPU (%)")
            axes[0].grid(alpha=0.4)

            axes[1].plot(ts, rss, color="#2E7D32", linewidth=1.0)
            axes[1].set_ylabel("RAM (MB)")
            axes[1].grid(alpha=0.4)

            axes[2].plot(ts, disk, color="#C62828", linewidth=1.0)
            axes[2].set_ylabel("Disk read (MB/s)")
            axes[2].set_xlabel("Time (s)")
            axes[2].grid(alpha=0.4)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            path = out_dir / f"05_system_resources_{loader}_{rt}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {path}")


def _plot_feature_pipeline_timelines(agg: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("active_batches_current", "Active batches"),
        ("active_samplers_current", "Samplers (non-idle)"),
        ("read_queue_current", "Read queue"),
        ("stitch_queue_current", "Stitch queue"),
    ]

    for loader in _ordered_loaders(agg):
        for rt in RUN_TYPES:
            if rt not in agg[loader]:
                continue
            timelines = agg[loader][rt].get("feature_pipeline_timelines", [])
            if not timelines:
                continue

            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f"Feature pipeline queues — {loader} / {rt}")

            for ax, (metric_key, metric_label) in zip(axes, metrics):
                has_data = False
                for idx, timeline in enumerate(timelines):
                    samples = timeline.get("samples", [])
                    if not samples:
                        continue
                    xs = [sample.get("timestamp_ms", 0) / 1000.0 for sample in samples]
                    ys = [sample.get(metric_key, 0) for sample in samples]
                    run_id = timeline.get("run_id")
                    label = f"run {run_id}" if run_id is not None else f"series {idx}"
                    ax.step(xs, ys, where="post", label=label)
                    has_data = True
                ax.set_ylabel(metric_label)
                ax.grid(axis="y", alpha=0.4)
                if has_data and len(timelines) > 1:
                    ax.legend(loc="upper right", fontsize=8)

            axes[-1].set_xlabel("Time (s)")
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            path = out_dir / f"04_feature_pipeline_queues_{loader}_{rt}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {path}")


def plot_results(agg: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = _ordered_loaders(agg)
    has_warm = any("warm" in agg[l] for l in loaders)
    run_label = "warm" if has_warm else "cold"

    # -- Box plot: batch time distribution (no outliers) ----------------------
    box_data, box_labels = [], []
    for loader in loaders:
        rt = _best_run_type(agg, loader)
        batches = _non_profiled(agg[loader][rt]["batches"], loader)
        box_data.append([b["total_ms"] for b in batches])
        box_labels.append(loader)

    if box_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=False)
        ax.set_ylabel("Batch time (ms)")
        ax.set_title(f"Batch time distribution — {run_label} runs (no outliers)")
        ax.grid(axis="y", alpha=0.4)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        p = out_dir / "03_batch_time_distribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p}")

    _plot_system_resources_timeseries(agg, out_dir)
    _plot_feature_pipeline_timelines(agg, out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "results", nargs="?", type=Path, default=None,
        help="Run directory or parent directory. Omit to auto-pick the latest run.",
    )
    ap.add_argument("--plots", action="store_true", help="Generate PNG plots.")
    args = ap.parse_args()

    results_path = _resolve_path(args)
    agg = load_results(results_path)
    if not agg:
        sys.exit("No loader results found.")

    out_dir = results_path / "plots"
    print(f"\nWriting output → {out_dir}")
    save_table_images(agg, out_dir)
    if args.plots:
        plot_results(agg, out_dir)


if __name__ == "__main__":
    main()

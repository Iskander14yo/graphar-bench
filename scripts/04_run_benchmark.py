#!/usr/bin/env python3
"""Stage 4 benchmark runner."""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import psutil
import pyarrow  # noqa: F401 - must precede graphar C extension
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))  # graphar-bench/ → enables `from benchmarks.xxx`
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_yaml import DEFAULT_PATH, BenchmarkConfig, load_config  # noqa: E402
import neo4j_service  # noqa: E402

import graphar as gar
from benchmarks.gar_loader import iter_batches as _gar_iter
from benchmarks.neo4j_loader import Neo4jNeighborLoader, iter_batches as _neo4j_iter
from benchmarks.pyg_loader import PyGNeighborLoader, iter_batches as _pyg_iter
from benchmarks.timings import BatchTimings, SystemSample
from graphar.ml.torch import GARNeighborLoader

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent


# ---------------------------------------------------------------------------
# Hardware fingerprint
# ---------------------------------------------------------------------------

def _hardware_info() -> dict:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    disk_type = "unknown"
    try:
        out = subprocess.run(
            ["lsblk", "-d", "-o", "NAME,ROTA"], capture_output=True, text=True, check=False
        ).stdout
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                disk_type = "ssd/nvme" if parts[1] == "0" else "hdd"
                break
    except Exception:
        pass

    return {
        "cpu_model": cpu_model,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "disk_type": disk_type,
    }


# ---------------------------------------------------------------------------
# Background system monitor
# ---------------------------------------------------------------------------

class _SystemMonitor:
    """Samples CPU %, RSS, and disk read throughput at ~1 s intervals."""

    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._samples: list[SystemSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[SystemSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return list(self._samples)

    def _run(self) -> None:
        prev_disk = psutil.disk_io_counters()
        prev_t = time.perf_counter()
        while not self._stop.wait(1.0):
            now = time.perf_counter()
            cur_disk = psutil.disk_io_counters()
            dt = max(now - prev_t, 1e-9)
            read_bytes = (
                (cur_disk.read_bytes - prev_disk.read_bytes)
                if cur_disk and prev_disk
                else 0
            )
            try:
                # cpu_percent is process-wide, summed over all threads and logical CPUs,
                # so it can legally exceed 100% × cpu_physical_cores on HT/SMT machines.
                cpu = self._proc.cpu_percent()
                rss = self._proc.memory_info().rss / 1e6
            except psutil.NoSuchProcess:
                break
            self._samples.append(SystemSample(
                timestamp_ms=int((now - self._t0) * 1000),
                cpu_pct=cpu,
                rss_mb=round(rss, 1),
                disk_read_mb_s=round(read_bytes / dt / 1e6, 2),
            ))
            prev_disk = cur_disk
            prev_t = now


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha(repo_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _drop_os_cache() -> None:
    """Drop OS page cache only — safe to call for any loader."""
    subprocess.run(
        ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"], check=True
    )


def _clear_caches(loader_name: str, config: BenchmarkConfig) -> None:
    """Drop OS page cache; optionally restart a backing database."""
    _drop_os_cache()
    if loader_name.startswith("neo4j"):
        n = config.neo4j
        neo4j_service.restart(n.uri, n.database)


def _chunk_manager_stats(loader) -> dict[str, int] | None:
    if not hasattr(loader, "chunk_manager_stats"):
        return None
    return loader.chunk_manager_stats()


def _stats_delta(after: dict[str, int] | None, before: dict[str, int] | None) -> dict[str, int] | None:
    if after is None or before is None:
        return None
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------

def _make_gar_loader(config: BenchmarkConfig) -> GARNeighborLoader:
    g = config.gar
    graph_info = gar.GraphInfo.load(str(Path(config.gar_graph_path).resolve()))
    features = [f"f{i:03d}" for i in range(config.num_features)]
    return GARNeighborLoader(
        graph_info,
        vertex_type=g.vertex_type,
        edge_type=g.edge_type,
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        features=features,
        ram_for_loader_mb=g.ram_for_loader_mb,
        num_workers=g.num_workers,
    )


def _make_neo4j_loader(config: BenchmarkConfig, loader_name: str) -> Neo4jNeighborLoader:
    n = config.neo4j
    strategy = "global" if loader_name == "neo4j-global" else "per_node"
    return Neo4jNeighborLoader(
        uri=n.uri,
        database=n.database,
        vertex_type="node",
        edge_type="edge",
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        features=config.features,
        num_features=config.num_features,
        profile_every_n=n.profile_every_n,
        strategy=strategy,
    )


def _make_pyg_loader(config: BenchmarkConfig) -> PyGNeighborLoader:
    return PyGNeighborLoader(
        dataset_name=config.dataset,
        ogb_root=config.ogb_root,
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_features=config.num_features,
    )


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def _run_epoch(
    loader, iter_fn, monitor: _SystemMonitor, desc: str, batch_limit: int | None
) -> tuple[list[BatchTimings], list[SystemSample], float]:
    """Returns (batch_timings, system_samples, epoch_time_ms)."""
    total = len(loader) # if hasattr(loader, "__len__") else None
    batches = iter_fn(loader)
    if batch_limit is not None:
        total = min(total, batch_limit)
        batches = itertools.islice(batches, batch_limit)
    monitor.start()
    batch_timings: list[BatchTimings] = []
    t0 = time.perf_counter()
    with tqdm(batches, total=total, desc=desc, unit="batch", leave=False) as pbar:
        for _batch, bt in pbar:
            batch_timings.append(bt)
            pbar.set_postfix({"ms": f"{bt.total_ms:.0f}"})
    epoch_time_ms = (time.perf_counter() - t0) * 1000
    system_samples = monitor.stop()
    return batch_timings, system_samples, epoch_time_ms


# ---------------------------------------------------------------------------
# Per-loader orchestration
# ---------------------------------------------------------------------------

def _run_loader(
    loader_name: str,
    make_loader: Callable[[], object],
    iter_fn,
    config: BenchmarkConfig,
    result_dir: Path,
) -> None:
    monitor = _SystemMonitor()
    runs = []
    loader = None

    try:
        for run_id in range(config.num_runs):
            run_type = "cold" if run_id == 0 else "warm"
            print(f"  [{loader_name}] run {run_id} ({run_type})...", flush=True)

            if run_type == "cold":
                try:
                    _clear_caches(loader_name, config)
                except Exception as e:
                    print(f"  WARNING: cache clear failed: {e}", flush=True)

            if loader is None:
                loader = make_loader()

            chunk_stats_before = _chunk_manager_stats(loader)
            batch_timings, system_samples, epoch_time_ms = _run_epoch(
                loader,
                iter_fn,
                monitor,
                desc=f"{loader_name}/{run_type}",
                batch_limit=config.batch_limit,
            )
            chunk_stats = _stats_delta(_chunk_manager_stats(loader), chunk_stats_before)
            mean_ms = (
                sum(bt.total_ms for bt in batch_timings) / len(batch_timings)
                if batch_timings else 0.0
            )
            print(f"    {len(batch_timings)} batches, mean={mean_ms:.1f} ms, epoch={epoch_time_ms/1000:.1f} s", flush=True)

            runs.append({
                "run_id": run_id,
                "type": run_type,
                "epoch_time_ms": epoch_time_ms,
                "batches": [dataclasses.asdict(bt) for bt in batch_timings],
                "system_metrics": [dataclasses.asdict(ss) for ss in system_samples],
            })
            if chunk_stats is not None:
                runs[-1]["chunk_manager"] = chunk_stats
    finally:
        close = getattr(loader, "close", None)
        if callable(close):
            close()
        if loader_name.startswith("neo4j"):
            neo4j_service.stop()

    out = result_dir / f"{loader_name}.json"
    out.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"  → {out}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(config: BenchmarkConfig, result_dir: Path | None = None) -> None:
    loaders_to_run = config.loaders
    dataset = config.dataset
    seed = config.seed

    torch.manual_seed(seed)

    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        result_dir = Path("graphar-bench/results") / dataset / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = Path(result_dir)
        timestamp = result_dir.name

    sha = _git_sha(_REPO_ROOT)
    bench_sha = _git_sha(_BENCH_ROOT)
    cfg_dump = dataclasses.asdict(config)
    run_info = {
        "timestamp": timestamp,
        "git_sha": sha,
        "graphar_bench_git_sha": bench_sha,
        "hardware": _hardware_info(),
        "notes": "",
        "config": cfg_dump,
    }
    (result_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Results dir: {result_dir}")

    for loader_name in loaders_to_run:
        print(f"\n=== {loader_name} ===", flush=True)
        try:
            if loader_name == "gar":
                make_loader = lambda: _make_gar_loader(config)
                iter_fn = _gar_iter
            elif loader_name in ("neo4j-global", "neo4j-per-node"):
                n = config.neo4j
                neo4j_service.ensure_running(n.uri, n.database)
                make_loader = lambda name=loader_name: _make_neo4j_loader(config, name)
                iter_fn = _neo4j_iter
            elif loader_name == "pyg-inmem":
                make_loader = lambda: _make_pyg_loader(config)
                iter_fn = _pyg_iter
            else:
                print(f"  Unknown loader '{loader_name}', skipping.")
                continue

            _run_loader(
                loader_name,
                make_loader,
                iter_fn,
                config,
                result_dir,
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    print(f"\nDone.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_PATH, help="Benchmark YAML.")
    p.add_argument("--result-dir", type=Path, default=None, help="Pre-created output directory.")
    args = p.parse_args()
    run_benchmark(load_config(args.config), result_dir=args.result_dir)


if __name__ == "__main__":
    main()

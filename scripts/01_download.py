#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from benchmark_yaml import DEFAULT_PATH, BenchmarkConfig, load_config  # noqa: E402


def _ogb_url(dataset: str) -> tuple[str, str]:
    """(url, download_name) from OGB's bundled master.csv."""
    import ogb

    master = Path(ogb.__file__).parent / "nodeproppred" / "master.csv"
    with open(master, newline="") as f:
        cols = next(csv.reader(f))
        idx = cols.index(dataset)
        meta = {r[0]: r[idx] for r in csv.reader(f)}
    return meta["url"], meta["download_name"]


def _npy_shape(zf: zipfile.ZipFile, entry: str) -> tuple[int, ...]:
    with zf.open(entry) as f:
        f.read(6)
        major, _ = struct.unpack("BB", f.read(2))
        hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
        hdr = f.read(hlen).decode("latin1")
    m = re.search(r"'shape'\s*:\s*\(([^)]*)\)", hdr)
    assert m, f"No shape in {entry}"
    return tuple(int(x) for x in m.group(1).split(",") if x.strip())


def download(config: BenchmarkConfig) -> None:
    root = Path(config.ogb_root)
    dataset_dir = root / config.dataset.replace("-", "_")
    raw_dir = dataset_dir / "raw"

    if not raw_dir.exists():
        url, name = _ogb_url(config.dataset)
        root.mkdir(parents=True, exist_ok=True)
        zip_path = root / f"{name}.zip"
        # -c resumes interrupted downloads
        subprocess.run(["wget", "-c", url, "-O", str(zip_path)], check=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        zip_path.unlink()
        # OGB expects ogbn_{name}/ layout; downloaded zip extracts to {download_name}/.
        shutil.move(str(root / name), str(dataset_dir))

    print(f"Dataset: {config.dataset}")
    npz = raw_dir / "data.npz"
    if npz.exists():
        with zipfile.ZipFile(npz) as zf:
            n_nodes, feat_dim = _npy_shape(zf, "node_feat.npy")
            _, n_edges = _npy_shape(zf, "edge_index.npy")
        print(f"Nodes: {n_nodes:,}  Edges: {n_edges:,}  Feat dim: {feat_dim}")


def main() -> None:
    p = argparse.ArgumentParser(description="Download OGB node-property dataset.")
    p.add_argument("--config", type=Path, default=DEFAULT_PATH, help="Benchmark YAML.")
    download(load_config(p.parse_args().config))


if __name__ == "__main__":
    main()

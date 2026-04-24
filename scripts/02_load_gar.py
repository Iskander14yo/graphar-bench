#!/usr/bin/env python3

from __future__ import annotations

import argparse
import functools
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_yaml import DEFAULT_PATH, BenchmarkConfig, load_config  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
_CPP_BINARY = _SCRIPTS.parent / "convert" / "build" / "ogbn_to_gar"


def _graph_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / f"{dataset}.graph.yml"


def _ogb_raw_dir(ogb_root: str, dataset: str) -> Path:
    return Path(ogb_root) / dataset.replace("-", "_") / "raw"


def _vertex_ipc_schema(feat_dim: int) -> pa.Schema:
    """Non-nullable fields so Parquet omits definition levels on read."""
    return pa.schema(
        [pa.field("id", pa.int64(), nullable=False)]
        + [pa.field(f"f{i:03d}", pa.float32(), nullable=False) for i in range(feat_dim)]
        + [pa.field("label", pa.int64(), nullable=False)]
    )


def _is_binary_ogb(raw_dir: Path) -> bool:
    return (raw_dir / "data.npz").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mark_complete(path: Path, n_rows: int) -> None:
    """Write side-car <file>.ok with row count after a successful write."""
    (path.parent / (path.name + ".ok")).write_text(str(n_rows))


def _is_complete(path: Path, expected_rows: int) -> bool:
    """True iff both the Arrow file and its .ok marker exist with the right count."""
    ok = path.parent / (path.name + ".ok")
    if not path.exists() or not ok.exists():
        return False
    try:
        return int(ok.read_text().strip()) == expected_rows
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Binary OGB format (e.g. ogbn-papers100M)
# data.npz:       node_feat (float32, N×F), edge_index (int64, 2×E), …
# node-label.npz: node_label (float32 or int64, N×1) — NaN = unlabeled
# ─────────────────────────────────────────────────────────────────────────────

def _read_npy_header(f) -> tuple[tuple[int, ...], np.dtype]:
    """Parse shape and dtype from an open npy stream (no seeking required)."""
    f.read(6)  # magic: \x93NUMPY
    major, _ = struct.unpack("BB", f.read(2))
    hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
    hdr = f.read(hlen).decode("latin1")
    shape_m = re.search(r"'shape'\s*:\s*\(([^)]*)\)", hdr)
    descr_m = re.search(r"'descr'\s*:\s*'([^']*)'", hdr)
    if shape_m is None or descr_m is None:
        raise ValueError(f"Unrecognised npy header: {hdr!r}")
    shape = tuple(int(x.strip()) for x in shape_m.group(1).split(",") if x.strip())
    dtype = np.dtype(descr_m.group(1))
    return shape, dtype


def _npz_array_shape(npz_path: Path, entry: str) -> tuple[int, ...]:
    """Read only the npy header from a compressed npz entry — no bulk decompression."""
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open(entry) as f:
            shape, _ = _read_npy_header(f)
    return shape


def _save_arrow_from_binary_ogb(
    raw_dir: Path,
    data_dir: Path,
    vertex_chunk_size: int,  # batch aligned to C++ chunk size — each Arrow batch = one chunk
    edge_batch: int,
) -> tuple[int, int]:
    """Stream binary OGB files to Arrow IPC without loading full arrays into RAM.

    Writes:
      vertices.arrow  — id (int64), f000..fFFF (float32), label (int64)
      edges_src.arrow — src_id (int64)
      edges_dst.arrow — dst_id (int64)

    edge_index is stored as (2, E) C-order in the npz: all src first, then all dst.
    We stream through the decompressor once: first half → edges_src, second → edges_dst.
    Peak RAM per step: max(labels ~888 MB, one vertex batch ~520 MB, one edge batch ~40 MB).
    """
    data_npz = raw_dir / "data.npz"
    label_npz = raw_dir / "node-label.npz"

    # Read shapes from npy headers only (tiny decompression, headers are a few bytes).
    feat_shape = _npz_array_shape(data_npz, "node_feat.npy")   # (N, F)
    ei_shape   = _npz_array_shape(data_npz, "edge_index.npy")  # (2, E)
    N, F = feat_shape[0], feat_shape[1]
    E = ei_shape[1]

    data_dir.mkdir(parents=True, exist_ok=True)
    vertices_path  = data_dir / "vertices.arrow"
    edges_src_path = data_dir / "edges_src.arrow"
    edges_dst_path = data_dir / "edges_dst.arrow"

    # ── Vertices ─────────────────────────────────────────────────────────────
    if _is_complete(vertices_path, N):
        print(f"vertices.arrow already complete ({N:,} rows), skipping.")
    else:
        # Labels come from a separate small file (~5 MB compressed → ~888 MB).
        print("Loading labels...")
        with zipfile.ZipFile(label_npz) as zf:
            with zf.open("node_label.npy") as f:
                l_shape, l_dtype = _read_npy_header(f)
                labels_raw = (
                    np.frombuffer(f.read(N * l_dtype.itemsize), dtype=l_dtype)
                    .reshape(l_shape).copy()
                )

        # Float32 labels with NaN mark unlabeled nodes (papers100M convention) → -1.
        flat = labels_raw.reshape(-1)
        if l_dtype == np.float32:
            labels_1d = np.where(np.isnan(flat), np.int64(-1), flat.astype(np.int64, copy=False))
        else:
            labels_1d = flat.astype(np.int64, copy=False)
        del labels_raw

        v_schema = _vertex_ipc_schema(F)
        row_bytes = F * 4  # node_feat is always float32 (verified from npy header)

        print(f"Streaming {N:,} vertices (F={F}) to vertices.arrow...")
        with zipfile.ZipFile(data_npz) as zf:
            with zf.open("node_feat.npy") as feat_f:
                feat_shape_hdr, feat_dtype = _read_npy_header(feat_f)
                row_bytes = F * feat_dtype.itemsize
                with pa_ipc.new_file(str(vertices_path), v_schema) as w:
                    start = 0
                    while start < N:
                        n = min(vertex_chunk_size, N - start)
                        chunk = (
                            np.frombuffer(feat_f.read(n * row_bytes), dtype=feat_dtype)
                            .reshape((n, F)).astype(np.float32, copy=False)
                        )
                        arrays = (
                            [pa.array(np.arange(start, start + n, dtype=np.int64))]
                            + [pa.array(chunk[:, i]) for i in range(F)]
                            + [pa.array(labels_1d[start : start + n])]
                        )
                        w.write_batch(
                            pa.record_batch(dict(zip(v_schema.names, arrays)), schema=v_schema)
                        )
                        start += n
                        print(f"  {start:,}/{N:,}", end="\r", flush=True)
                print()
        del labels_1d
        _mark_complete(vertices_path, N)
        print(f"vertices.arrow done ({N:,} rows).")

    # ── Edges ─────────────────────────────────────────────────────────────────
    # edge_index shape (2, E) in C-order: first E×8 bytes = all src, then E×8 bytes = all dst.
    # One decompression pass through the zip entry streams both halves sequentially.
    if _is_complete(edges_src_path, E) and _is_complete(edges_dst_path, E):
        print(f"Edge files already complete ({E:,} edges), skipping.")
    else:
        src_schema = pa.schema([pa.field("src_id", pa.int64(), nullable=False)])
        dst_schema = pa.schema([pa.field("dst_id", pa.int64(), nullable=False)])

        print(f"Streaming {E:,} edges to edges_src.arrow + edges_dst.arrow...")
        with zipfile.ZipFile(data_npz) as zf:
            with zf.open("edge_index.npy") as f:
                _, ei_dtype = _read_npy_header(f)
                itemsize = ei_dtype.itemsize
                chunk_bytes = edge_batch * itemsize

                # First E elements in stream = src (row 0 of the (2,E) array).
                with pa_ipc.new_file(str(edges_src_path), src_schema) as sw:
                    remaining = E * itemsize
                    done = 0
                    while remaining > 0:
                        n_bytes = min(chunk_bytes, remaining)
                        raw = f.read(n_bytes)
                        arr = np.frombuffer(raw, dtype=ei_dtype)
                        sw.write_batch(
                            pa.record_batch({"src_id": pa.array(arr, type=pa.int64())},
                                            schema=src_schema)
                        )
                        done += len(arr)
                        remaining -= n_bytes
                        print(f"  src {done:,}/{E:,}", end="\r", flush=True)
                print()

                # Next E elements = dst (row 1 of the (2,E) array).
                with pa_ipc.new_file(str(edges_dst_path), dst_schema) as dw:
                    remaining = E * itemsize
                    done = 0
                    while remaining > 0:
                        n_bytes = min(chunk_bytes, remaining)
                        raw = f.read(n_bytes)
                        arr = np.frombuffer(raw, dtype=ei_dtype)
                        dw.write_batch(
                            pa.record_batch({"dst_id": pa.array(arr, type=pa.int64())},
                                            schema=dst_schema)
                        )
                        done += len(arr)
                        remaining -= n_bytes
                        print(f"  dst {done:,}/{E:,}", end="\r", flush=True)
                print()

        _mark_complete(edges_src_path, E)
        _mark_complete(edges_dst_path, E)
        print(f"Edge files done ({E:,} edges).")

    return N, F


# ─────────────────────────────────────────────────────────────────────────────
# CSV OGB format (e.g. ogbn-arxiv, ogbn-products)
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _ogb_meta(dataset: str) -> dict[str, object]:
    import ogb.nodeproppred as ogb_nodeproppred

    master_path = Path(ogb_nodeproppred.__file__).resolve().parent / "master.csv"
    meta = pd.read_csv(master_path, index_col=0).T
    if dataset not in meta.index:
        raise KeyError(f"Dataset metadata not found in OGB master.csv: {dataset}")
    return meta.loc[dataset].to_dict()


def _csv_count(raw_dir: Path, name: str) -> int:
    counts = pd.read_csv(raw_dir / name, compression="gzip", header=None, dtype=np.int64)
    return int(counts.iloc[:, 0].sum())


def _save_arrow_from_csv_ogb(
    raw_dir: Path,
    data_dir: Path,
    vertex_batch: int,
    edge_batch: int,
    add_inverse_edge: bool,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    feat_path = raw_dir / "node-feat.csv.gz"
    label_path = raw_dir / "node-label.csv.gz"
    edge_path = raw_dir / "edge.csv.gz"

    N = _csv_count(raw_dir, "num-node-list.csv.gz")
    E = _csv_count(raw_dir, "num-edge-list.csv.gz")
    feat_probe = pd.read_csv(feat_path, compression="gzip", header=None, nrows=1)
    if feat_probe.empty:
        raise ValueError(f"node-feat.csv.gz is empty: {feat_path}")
    F = feat_probe.shape[1]

    vertices_path = data_dir / "vertices.arrow"
    edges_src_path = data_dir / "edges_src.arrow"
    edges_dst_path = data_dir / "edges_dst.arrow"

    if _is_complete(vertices_path, N):
        print(f"vertices.arrow already complete ({N:,} rows), skipping.")
    else:
        v_schema = _vertex_ipc_schema(F)
        feat_iter = pd.read_csv(
            feat_path,
            compression="gzip",
            header=None,
            dtype=np.float32,
            chunksize=vertex_batch,
        )
        label_iter = pd.read_csv(
            label_path,
            compression="gzip",
            header=None,
            chunksize=vertex_batch,
        )
        print(f"Streaming {N:,} vertices (F={F}) to vertices.arrow...")
        with pa_ipc.new_file(str(vertices_path), v_schema) as w:
            start = 0
            for feat_chunk, label_chunk in zip(feat_iter, label_iter, strict=True):
                chunk = feat_chunk.to_numpy(dtype=np.float32, copy=False)
                labels_1d = label_chunk.to_numpy().reshape(-1).astype(np.int64, copy=False)
                if chunk.ndim != 2 or chunk.shape[1] != F:
                    raise ValueError(f"Expected node_feat chunk shape (*, {F}), got {chunk.shape}")
                if chunk.shape[0] != labels_1d.shape[0]:
                    raise ValueError(
                        f"Feature/label chunk row mismatch: {chunk.shape[0]} vs {labels_1d.shape[0]}"
                    )
                end = start + chunk.shape[0]
                arrays = (
                    [pa.array(np.arange(start, end, dtype=np.int64))]
                    + [pa.array(chunk[:, i]) for i in range(F)]
                    + [pa.array(labels_1d)]
                )
                w.write_batch(pa.record_batch(dict(zip(v_schema.names, arrays)), schema=v_schema))
                start = end
                print(f"  {start:,}/{N:,}", end="\r", flush=True)
        print()
        if start != N:
            raise ValueError(f"Expected {N} vertices, wrote {start}")
        _mark_complete(vertices_path, N)
        print(f"vertices.arrow done ({N:,} rows).")

    total_edges = E * (2 if add_inverse_edge else 1)
    if _is_complete(edges_src_path, total_edges) and _is_complete(edges_dst_path, total_edges):
        print(f"Edge files already complete ({total_edges:,} edges), skipping.")
        return

    src_schema = pa.schema([pa.field("src_id", pa.int64(), nullable=False)])
    dst_schema = pa.schema([pa.field("dst_id", pa.int64(), nullable=False)])
    edge_iter = pd.read_csv(
        edge_path,
        compression="gzip",
        header=None,
        dtype=np.int64,
        chunksize=edge_batch,
    )
    print(
        f"Streaming {E:,} edges to edges_src.arrow + edges_dst.arrow"
        + (" with inverse edges..." if add_inverse_edge else "...")
    )
    with pa_ipc.new_file(str(edges_src_path), src_schema) as sw:
        with pa_ipc.new_file(str(edges_dst_path), dst_schema) as dw:
            done = 0
            for edge_chunk in edge_iter:
                chunk = edge_chunk.to_numpy(dtype=np.int64, copy=False)
                if chunk.ndim != 2 or chunk.shape[1] != 2:
                    raise ValueError(f"Expected edge chunk shape (*, 2), got {chunk.shape}")
                src = chunk[:, 0]
                dst = chunk[:, 1]
                sw.write_batch(
                    pa.record_batch({"src_id": pa.array(src, type=pa.int64())}, schema=src_schema)
                )
                dw.write_batch(
                    pa.record_batch({"dst_id": pa.array(dst, type=pa.int64())}, schema=dst_schema)
                )
                if add_inverse_edge:
                    sw.write_batch(
                        pa.record_batch({"src_id": pa.array(dst, type=pa.int64())}, schema=src_schema)
                    )
                    dw.write_batch(
                        pa.record_batch({"dst_id": pa.array(src, type=pa.int64())}, schema=dst_schema)
                    )
                done += chunk.shape[0]
                written = done * (2 if add_inverse_edge else 1)
                print(f"  {written:,}/{total_edges:,}", end="\r", flush=True)
    print()
    if done != E:
        raise ValueError(f"Expected {E} edges, wrote {done}")
    _mark_complete(edges_src_path, total_edges)
    _mark_complete(edges_dst_path, total_edges)
    print(f"Edge files done ({total_edges:,} edges).")


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_ogb_to_gar(config: BenchmarkConfig) -> None:
    if not _CPP_BINARY.exists():
        raise FileNotFoundError(
            f"C++ converter binary not found: {_CPP_BINARY}\n"
            "Build it first:\n"
            "  mkdir -p graphar-bench/convert/build\n"
            "  cd graphar-bench/convert/build\n"
            "  cmake .. && make -j$(nproc)"
        )

    g = config.gar
    dataset = config.dataset
    output_dir = Path(config.gar_root) / dataset
    graph_yml = _graph_path(output_dir, dataset)
    data_dir = Path(g.tmp_root) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = _ogb_raw_dir(config.ogb_root, dataset)

    extra_args: list[str] = []

    if _is_binary_ogb(raw_dir):
        print(f"Binary OGB format detected for {dataset}, streaming from raw files.")
        num_vertices, feat_dim = _save_arrow_from_binary_ogb(
            raw_dir=raw_dir,
            data_dir=data_dir,
            # Align batch to chunk size so each Arrow batch = exactly one C++ vertex chunk.
            vertex_chunk_size=g.vertex_chunk_size,
            edge_batch=g.edge_write_batch_size,
        )
        # Pass known dimensions so C++ can stream vertices without reading the full table.
        extra_args = ["--num-vertices", str(num_vertices), "--feat-dim", str(feat_dim)]
    else:
        meta = _ogb_meta(dataset)
        if str(meta.get("has_edge_attr", "")).lower() == "true":
            raise NotImplementedError(f"Streaming CSV edge attributes are not supported for {dataset}")
        for key in ("additional node files", "additional edge files"):
            value = str(meta.get(key, "")).lower()
            if value not in {"", "none", "nan"}:
                raise NotImplementedError(
                    f"Streaming CSV auxiliary files are not supported for {dataset}: {key}"
                )

        print(f"CSV OGB format detected for {dataset}, streaming from raw files.")
        _save_arrow_from_csv_ogb(
            raw_dir=raw_dir,
            data_dir=data_dir,
            vertex_batch=g.vertex_write_batch_size,
            edge_batch=g.edge_write_batch_size,
            add_inverse_edge=str(meta.get("add_inverse_edge", "")).lower() == "true",
        )

    cmd = [
        str(_CPP_BINARY),
        "--output-dir", str(output_dir.resolve()),
        "--name", dataset,
        "--data-dir", str(data_dir.resolve()),
        "--vertex-chunk", str(g.vertex_chunk_size),
        "--edge-chunk", str(g.edge_chunk_size),
        *extra_args,
    ]
    print("Running C++ converter:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not graph_yml.exists():
        raise FileNotFoundError(f"GAR conversion completed but graph file not found: {graph_yml}")

    print(f"Converted dataset to GAR: {graph_yml}")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert OGB dataset into GAR format.")
    p.add_argument("--config", type=Path, default=DEFAULT_PATH, help="Benchmark YAML.")
    convert_ogb_to_gar(load_config(p.parse_args().config))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_yaml import DEFAULT_PATH, BenchmarkConfig, load_config  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
_CPP_BINARY = _SCRIPTS.parent / "convert" / "build" / "ogbn_to_gar"


def _graph_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / f"{dataset}.graph.yml"


def _load_ogb_graph(dataset: str, root: str) -> tuple[dict, np.ndarray]:
    from ogb.nodeproppred import NodePropPredDataset

    try:
        ogb = NodePropPredDataset(name=dataset, root=root)
        graph, labels_raw = ogb[0]
        return graph, np.asarray(labels_raw)
    except zipfile.BadZipFile:
        root_path = Path(root)
        for zip_path in root_path.rglob("*.zip"):
            zip_path.unlink(missing_ok=True)
        ogb = NodePropPredDataset(name=dataset, root=root)
        graph, labels_raw = ogb[0]
        return graph, np.asarray(labels_raw)


def _save_arrow(
    data_dir: Path,
    node_feat: np.ndarray,
    labels: np.ndarray,
    edge_index: np.ndarray,
    vertex_batch: int,
    edge_batch: int,
) -> None:
    """Write Arrow IPC files consumed by the C++ converter."""
    data_dir.mkdir(parents=True, exist_ok=True)

    N, F = node_feat.shape
    labels_1d = labels.reshape(-1).astype(np.int64, copy=False)

    # vertices.arrow: id, f000..fFFF, label
    v_schema = pa.schema(
        [("id", pa.int64())]
        + [(f"f{i:03d}", pa.float32()) for i in range(F)]
        + [("label", pa.int64())]
    )
    with pa_ipc.new_file(str(data_dir / "vertices.arrow"), v_schema) as w:
        for start in range(0, N, vertex_batch):
            end = min(start + vertex_batch, N)
            chunk = node_feat[start:end]
            arrays = (
                [pa.array(np.arange(start, end, dtype=np.int64))]
                + [pa.array(chunk[:, i]) for i in range(F)]
                + [pa.array(labels_1d[start:end])]
            )
            w.write_batch(pa.record_batch(dict(zip(v_schema.names, arrays)),
                                          schema=v_schema))

    # edges.arrow: src_id, dst_id
    E = edge_index.shape[1]
    e_schema = pa.schema([("src_id", pa.int64()), ("dst_id", pa.int64())])
    with pa_ipc.new_file(str(data_dir / "edges.arrow"), e_schema) as w:
        for start in range(0, E, edge_batch):
            end = min(start + edge_batch, E)
            w.write_batch(pa.record_batch({
                "src_id": pa.array(edge_index[0, start:end], type=pa.int64()),
                "dst_id": pa.array(edge_index[1, start:end], type=pa.int64()),
            }, schema=e_schema))


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

    graph, labels = _load_ogb_graph(dataset, config.ogb_root)
    node_feat: np.ndarray = graph["node_feat"]
    edge_index: np.ndarray = graph["edge_index"]

    if node_feat.ndim != 2:
        msg = f"Expected node_feat to be 2D, got shape={node_feat.shape}"
        raise ValueError(msg)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = f"Expected edge_index shape (2, E), got shape={edge_index.shape}"
        raise ValueError(msg)

    data_dir = Path(g.tmp_root) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Writing Arrow IPC files for C++ converter...")
    _save_arrow(
        data_dir=data_dir,
        node_feat=node_feat,
        labels=labels,
        edge_index=edge_index,
        vertex_batch=g.vertex_write_batch_size,
        edge_batch=g.edge_write_batch_size,
    )

    cmd = [
        str(_CPP_BINARY),
        "--output-dir", str(output_dir.resolve()),
        "--name", dataset,
        "--data-dir", str(data_dir.resolve()),
        "--vertex-chunk", str(g.vertex_chunk_size),
        "--edge-chunk", str(g.edge_chunk_size),
    ]
    print("Running C++ converter:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not graph_yml.exists():
        msg = f"GAR conversion completed but graph file not found: {graph_yml}"
        raise FileNotFoundError(msg)

    print(f"Converted dataset to GAR: {graph_yml}")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert OGB dataset into GAR format.")
    p.add_argument("--config", type=Path, default=DEFAULT_PATH, help="Benchmark YAML.")
    convert_ogb_to_gar(load_config(p.parse_args().config))


if __name__ == "__main__":
    main()

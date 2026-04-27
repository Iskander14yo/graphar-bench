from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_EXPS = Path(__file__).resolve().parent.parent
DEFAULT_PATH = _EXPS / "config" / "benchmark.yaml"


@dataclass(frozen=True)
class GarSection:
    vertex_type: str
    edge_type: str
    tmp_root: str
    vertex_chunk_size: int
    edge_chunk_size: int
    vertex_write_batch_size: int
    edge_write_batch_size: int
    ram_for_loader_mb: int = 0
    num_workers: int = 4
    feature_cursor_count: int = 1
    feature_cursor_trail_chunks: int = 10


@dataclass(frozen=True)
class Neo4jSection:
    uri: str
    database: str
    profile_every_n: int | None
    csv_root: str
    skip_import: bool
    node_batch_size: int
    edge_batch_size: int
    force: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str
    loaders: list[str]
    batch_size: int
    num_neighbors: list[int]
    features: list[str]
    num_features: int
    num_runs: int
    shuffle: bool
    seed: int
    ogb_root: str
    gar_root: str
    gar: GarSection
    neo4j: Neo4jSection
    batch_limit: int | None = None

    @property
    def gar_graph_path(self) -> str:
        return str(Path(self.gar_root) / self.dataset / f"{self.dataset}.graph.yml")


def load_config(path: Path | str | None = None) -> BenchmarkConfig:
    p = Path(path) if path else DEFAULT_PATH
    raw = yaml.safe_load(p.read_text())
    batch_limit = raw.get("batch_limit")
    if batch_limit is not None and batch_limit < 0:
        raise ValueError("batch_limit must be non-negative or null")
    return BenchmarkConfig(
        dataset=raw["dataset"],
        loaders=list(raw["loaders"]),
        batch_size=raw["batch_size"],
        num_neighbors=list(raw["num_neighbors"]),
        features=list(raw["features"]),
        num_features=raw["num_features"],
        num_runs=raw["num_runs"],
        shuffle=raw["shuffle"],
        seed=raw["seed"],
        ogb_root=raw["ogb_root"],
        gar_root=raw["gar_root"],
        gar=GarSection(**raw["gar"]),
        neo4j=Neo4jSection(**raw["neo4j"]),
        batch_limit=batch_limit,
    )

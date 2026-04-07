from __future__ import annotations

import time
from dataclasses import dataclass, field


class _Timer:
    """Context manager that appends elapsed seconds to store[name]."""

    __slots__ = ("_name", "_store", "_t")

    def __init__(self, name: str, store: dict) -> None:
        self._name = name
        self._store = store

    def __enter__(self) -> _Timer:
        self._t = time.perf_counter()
        return self

    def __exit__(self, exc_type, *_) -> None:
        if exc_type is None:
            self._store[self._name].append(time.perf_counter() - self._t)


@dataclass
class BatchTimings:
    batch_id: int
    total_ms: float
    retrieval_ms: float
    conversion_ms: float
    sampled_nodes: int
    sampled_edges: int
    # GAR-specific sub-stages (None for other loaders)
    sampling_ms: float | None = field(default=None)
    feature_fetch_ms: float | None = field(default=None)
    # Neo4j-specific (None for other loaders; populated every profile_every_n batches)
    neo4j_profile: dict | None = field(default=None)

@dataclass
class SystemSample:
    timestamp_ms: int       # ms since epoch start
    cpu_pct: float          # process CPU %, summed over all threads/logical CPUs; can exceed 100% × physical_cores
    rss_mb: float           # process RSS
    disk_read_mb_s: float   # system-wide disk read throughput

from __future__ import annotations

from collections.abc import Iterator

from torch_geometric.data import Data

from graphar.ml.torch import GARNeighborLoader

from .timings import BatchTimings


def iter_batches(loader: GARNeighborLoader) -> Iterator[tuple[Data, BatchTimings]]:
    for batch_id, (batch, prof) in enumerate(loader.profile()):
        yield batch, BatchTimings(
            batch_id=batch_id,
            total_ms=prof.total_ms,
            retrieval_ms=prof.sampling_ms + prof.feature_fetch_ms,
            conversion_ms=prof.conversion_ms,
            sampling_ms=prof.sampling_ms,
            feature_fetch_ms=prof.feature_fetch_ms,
            sampled_nodes=int(batch.n_id.size(0)),
            sampled_edges=int(batch.edge_index.size(1)) if batch.edge_index is not None else 0,
        )

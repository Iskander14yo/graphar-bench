from __future__ import annotations

from collections.abc import Callable

STAT_BUCKET_KEYS = (
    "chunk_manager",
    "feature_chunk_manager",
    "feature_pipeline",
    "feature_cursor",
)


def init_bucket(bucket: dict) -> None:
    for key in STAT_BUCKET_KEYS:
        bucket.setdefault(key, [])


def append_run(bucket: dict, run: dict) -> None:
    for key in STAT_BUCKET_KEYS:
        if key in run:
            bucket[key].append(run[key])


def _sum_entries(entries: list[dict], keys: tuple[str, ...]) -> dict[str, int]:
    return {
        key: sum(int(entry.get(key, 0)) for entry in entries)
        for key in keys
    }


def _manager_rows(
    entries: list[dict],
    run_type: str,
    fmt: Callable[[float, str], str],
) -> list[str] | None:
    if not entries:
        return None
    totals = _sum_entries(
        entries,
        (
            "requests",
            "leaders",
            "waiters",
            "completed",
            "failed",
            "ram_cache_hits",
            "ram_cache_misses",
            "ram_cache_evictions",
        ),
    )
    ram_cache_bytes = max(int(entry.get("ram_cache_bytes", 0)) for entry in entries)
    requests = totals["requests"]
    leaders = totals["leaders"]
    waiters = totals["waiters"]
    failed = totals["failed"]
    ram_hits = totals["ram_cache_hits"]
    ram_misses = totals["ram_cache_misses"]
    dedup_ratio = requests / leaders if leaders else float("nan")
    waiter_rate = waiters / requests if requests else float("nan")
    failure_rate = failed / requests if requests else float("nan")
    ram_total = ram_hits + ram_misses
    ram_hit_rate = ram_hits / ram_total if ram_total else float("nan")
    return [
        run_type,
        str(totals["requests"]),
        str(totals["leaders"]),
        str(totals["waiters"]),
        str(totals["completed"]),
        str(totals["failed"]),
        fmt(dedup_ratio, ".2f"),
        fmt(waiter_rate, ".2%"),
        fmt(failure_rate, ".2%"),
        str(ram_hits),
        str(ram_misses),
        fmt(ram_hit_rate, ".2%"),
        str(totals["ram_cache_evictions"]),
        fmt(ram_cache_bytes / 1e6),
    ]


def data_chunk_manager(
    agg: dict,
    ordered_loaders: list[str],
    run_types: list[str],
    fmt: Callable[[float, str], str],
) -> tuple[list[str], list[list[str]]]:
    headers = [
        "run",
        "Requests", "Leaders", "Waiters", "Completed", "Failed",
        "Dedup ratio", "Waiter rate", "Failure rate",
        "RAM hits", "RAM misses", "RAM hit rate", "RAM evict", "RAM MB",
    ]
    rows = []
    for loader in ordered_loaders:
        for run_type in run_types:
            if run_type not in agg[loader]:
                continue
            row = _manager_rows(
                agg[loader][run_type].get("chunk_manager", []),
                run_type,
                fmt,
            )
            if row is not None:
                rows.append(row)
    return headers, rows


def data_feature_chunk_manager(
    agg: dict,
    ordered_loaders: list[str],
    run_types: list[str],
    fmt: Callable[[float, str], str],
) -> tuple[list[str], list[list[str]]]:
    headers = [
        "run",
        "Requests", "Leaders", "Waiters", "Completed", "Failed",
        "Dedup ratio", "Waiter rate", "Failure rate",
        "RAM hits", "RAM misses", "RAM hit rate", "RAM evict", "RAM MB",
    ]
    rows = []
    for loader in ordered_loaders:
        if loader != "gar":
            continue
        for run_type in run_types:
            if run_type not in agg[loader]:
                continue
            row = _manager_rows(
                agg[loader][run_type].get("feature_chunk_manager", []),
                run_type,
                fmt,
            )
            if row is not None:
                rows.append(row)
    return headers, rows


def data_feature_cursor(
    agg: dict,
    ordered_loaders: list[str],
    run_types: list[str],
    fmt: Callable[[float, str], str],
    non_profiled: Callable[[list[dict], str], list[dict]],
) -> tuple[list[str], list[list[str]]]:
    headers = [
        "run",
        "cursor_count",
        "requests",
        "chunks_read",
        "chunks_served",
        "chunks/req",
        "batches/chunk",
        "rows_served",
        "rows/req",
        "trail_hit_rate",
        "avg_wait_ms",
        "wait_ms_max",
        "avg_service_ms",
        "service_ms_sum",
        "req_overhang",
    ]
    rows = []
    for loader in ordered_loaders:
        if loader != "gar":
            continue
        for run_type in run_types:
            if run_type not in agg[loader]:
                continue
            entries = agg[loader][run_type].get("feature_cursor", [])
            if not entries:
                continue
            totals = _sum_entries(
                entries,
                (
                    "requests",
                    "chunks_read",
                    "chunks_served",
                    "rows_served",
                    "batches_served",
                    "trail_hits",
                    "trail_misses",
                    "wait_ms_sum",
                    "service_ms_sum",
                ),
            )
            cursor_count = max(int(entry.get("cursor_count", 0)) for entry in entries)
            wait_ms_max = max(int(entry.get("wait_ms_max", 0)) for entry in entries)
            trail_total = totals["trail_hits"] + totals["trail_misses"]
            trail_hit_rate = (
                totals["trail_hits"] / trail_total if trail_total else float("nan")
            )
            requests = totals["requests"]
            chunks_read = totals["chunks_read"]
            chunks_served = totals["chunks_served"]
            batches_served = totals["batches_served"]
            rows_served = totals["rows_served"]
            avg_chunks_per_request = chunks_read / requests if requests else float("nan")
            avg_batches_per_chunk = (
                batches_served / chunks_read if chunks_read else float("nan")
            )
            avg_rows_per_request = rows_served / requests if requests else float("nan")
            avg_wait_ms = totals["wait_ms_sum"] / requests if requests else float("nan")
            avg_service_ms = (
                totals["service_ms_sum"] / chunks_served
                if chunks_served
                else float("nan")
            )
            consumed_batches = len(non_profiled(agg[loader][run_type]["batches"], loader))
            request_overhang = requests - consumed_batches
            rows.append([
                run_type,
                str(cursor_count),
                str(requests),
                str(chunks_read),
                str(chunks_served),
                fmt(avg_chunks_per_request, ".1f"),
                fmt(avg_batches_per_chunk, ".2f"),
                str(rows_served),
                fmt(avg_rows_per_request, ".0f"),
                fmt(trail_hit_rate, ".2%"),
                fmt(avg_wait_ms, ".0f"),
                str(wait_ms_max),
                fmt(avg_service_ms, ".1f"),
                str(totals["service_ms_sum"]),
                str(request_overhang),
            ])
    return headers, rows


def data_feature_pipeline(
    agg: dict,
    ordered_loaders: list[str],
    run_types: list[str],
    fmt: Callable[[float, str], str],
) -> tuple[list[str], list[list[str]]]:
    headers = [
        "run",
        "submitted",
        "completed",
        "pending_peak",
        "chunk_peak",
        "subscriptions",
        "chunk_reads",
        "chunk_reuses",
        "reuse_ratio",
        "stitch_tasks",
        "avg_stitch_wait_ms",
        "avg_stitch_service_ms",
    ]
    rows = []
    for loader in ordered_loaders:
        if loader != "gar":
            continue
        for run_type in run_types:
            if run_type not in agg[loader]:
                continue
            entries = agg[loader][run_type].get("feature_pipeline", [])
            if not entries:
                continue
            totals = _sum_entries(
                entries,
                (
                    "submitted_batches",
                    "completed_batches",
                    "chunk_subscriptions",
                    "chunk_reads",
                    "chunk_reuses",
                    "stitch_tasks",
                    "stitch_wait_ms_sum",
                    "stitch_service_ms_sum",
                ),
            )
            pending_peak = max(int(entry.get("pending_batches_peak", 0)) for entry in entries)
            chunk_peak = max(int(entry.get("active_chunk_keys_peak", 0)) for entry in entries)
            stitch_tasks = totals["stitch_tasks"]
            chunk_reads = totals["chunk_reads"]
            reuse_ratio = (
                totals["chunk_subscriptions"] / chunk_reads
                if chunk_reads
                else float("nan")
            )
            avg_stitch_wait_ms = (
                totals["stitch_wait_ms_sum"] / stitch_tasks
                if stitch_tasks
                else float("nan")
            )
            avg_stitch_service_ms = (
                totals["stitch_service_ms_sum"] / stitch_tasks
                if stitch_tasks
                else float("nan")
            )
            rows.append([
                run_type,
                str(totals["submitted_batches"]),
                str(totals["completed_batches"]),
                str(pending_peak),
                str(chunk_peak),
                str(totals["chunk_subscriptions"]),
                str(chunk_reads),
                str(totals["chunk_reuses"]),
                fmt(reuse_ratio, ".2f"),
                str(stitch_tasks),
                fmt(avg_stitch_wait_ms, ".1f"),
                fmt(avg_stitch_service_ms, ".1f"),
            ])
    return headers, rows

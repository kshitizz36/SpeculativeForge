from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

from speculate_forge.models import SpeculateConfig


def dispatch_parallel(
    configs: Iterable[SpeculateConfig],
    *,
    run_config: Callable[[SpeculateConfig], dict | None],
    max_workers: int = 3,
) -> list[tuple[SpeculateConfig, dict | None]]:
    """Fan out config runs in parallel using local threads.

    The Modal calls inside `run_config` are blocking network operations, so a
    small thread pool is enough to exercise the Phase C scheduler path without
    rewriting the whole environment async.
    """

    materialized = list(configs)
    if not materialized:
        return []

    results: list[tuple[SpeculateConfig, dict | None]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(materialized))) as pool:
        future_to_config = {
            pool.submit(run_config, config): config for config in materialized
        }
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                payload = future.result()
            except Exception as exc:  # pragma: no cover - defensive path
                payload = {"ready": False, "error": str(exc)}
            results.append((config, payload))
    return results

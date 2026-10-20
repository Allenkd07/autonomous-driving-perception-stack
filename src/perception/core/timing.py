"""Per-stage wall-clock timing, used by the pipeline and the benchmarks.

GPU work is asynchronous: backends must synchronise (e.g. torch.cuda.synchronize())
before returning, or the time is attributed to whichever stage syncs next.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.timings_ms[stage] = self.timings_ms.get(stage, 0.0) + elapsed_ms

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())

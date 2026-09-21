"""Simple timing context manager."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer(label: str = "") -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    prefix = f"{label}: " if label else ""
    print(f"{prefix}{elapsed:.4f}s")

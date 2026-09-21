"""Dataset scaling experiment placeholder.

Run multiple baseline trainings with different sample counts and
collect final validation metrics. Full orchestration is left to the
CLI / scripts layer to keep this module lightweight.
"""

from __future__ import annotations

from typing import Any


def scaling_schedule() -> list[int]:
    """Recommended dataset sizes for a scaling study."""
    return [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]

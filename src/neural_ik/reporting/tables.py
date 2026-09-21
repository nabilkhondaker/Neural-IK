"""Table formatting helpers for reports."""

from __future__ import annotations

from typing import Any


def format_metric_table(stats: dict[str, float], precision: int = 6) -> str:
    lines = ["| Metric | Value |", "|--------|-------|"]
    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.{precision}g} |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines)

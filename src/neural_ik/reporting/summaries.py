"""Short summary builders."""

from __future__ import annotations

from typing import Any


def one_line_summary(report: dict[str, Any]) -> str:
    name = report.get("name", "solver")
    stats = report.get("error_stats", {})
    mae = stats.get("mae", float("nan"))
    return f"{name}: MAE={mae:.6g} (n={stats.get('n', '?')})"

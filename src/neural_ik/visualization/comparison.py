"""Side-by-side solver comparison plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_solver_comparison(
    reports: list[dict[str, Any]],
    metric: str = "mae",
    ax: plt.Axes | None = None,
    title: str = "Solver Comparison",
) -> plt.Axes:
    names = [r["name"] for r in reports]
    values = [r["error_stats"].get(metric, float("nan")) for r in reports]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values, color=["#4472C4", "#70AD47", "#ED7D31"][: len(names)])
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4g}", ha="center", va="bottom", fontsize=8)
    return ax

"""Training curve plots."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt


def plot_training_curves(
    train_loss: Sequence[float],
    val_loss: Sequence[float] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Training Curves",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="train", color="#4472C4")
    if val_loss is not None and len(val_loss) > 0:
        ax.plot(range(1, len(val_loss) + 1), val_loss, label="val", color="#ED7D31")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax

"""Training callbacks (extensible hooks)."""

from __future__ import annotations

from typing import Any, Protocol


class Callback(Protocol):
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None: ...


class EarlyStopping:
    """Stop when monitored metric stops improving."""

    def __init__(self, patience: int = 10, mode: str = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        value = logs.get("val_loss")
        if value is None:
            return
        if self.best is None:
            self.best = value
            return
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

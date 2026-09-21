"""Model checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    epoch: int | None = None,
    val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    if epoch is not None:
        payload["epoch"] = epoch
    if val_loss is not None:
        payload["val_loss"] = val_loss
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload

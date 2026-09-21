"""PyTorch Dataset wrappers and I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from neural_ik.data.generation import IKData


class IKDataset(Dataset):
    """Map-style dataset of (position, joint) pairs."""

    def __init__(
        self,
        positions: NDArray[np.floating[Any]],
        joints: NDArray[np.floating[Any]],
        normalizer: Any | None = None,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float64)
        self.joints = np.asarray(joints, dtype=np.float64)
        self.normalizer = normalizer
        if normalizer is not None:
            self.positions = normalizer.transform_positions(self.positions)
            self.joints = normalizer.transform_joints(self.joints)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.positions[idx].astype(np.float32))
        y = torch.from_numpy(self.joints[idx].astype(np.float32))
        return x, y


def save_dataset(splits: dict[str, IKData], path: str | Path) -> None:
    """Save train/val/test splits to a single .npz archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for name, data in splits.items():
        payload[f"{name}_positions"] = data.positions
        payload[f"{name}_joints"] = data.joints
        if data.branch_labels is not None:
            payload[f"{name}_branch"] = data.branch_labels
        payload[f"{name}_meta"] = data.metadata
    np.savez_compressed(path, **payload)


def load_dataset(path: str | Path) -> dict[str, IKData]:
    """Load splits previously written by save_dataset."""
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    splits: dict[str, IKData] = {}
    for name in ("train", "val", "test"):
        if f"{name}_positions" not in z:
            continue
        meta = z[f"{name}_meta"].item() if f"{name}_meta" in z else {}
        branch = z[f"{name}_branch"] if f"{name}_branch" in z else None
        splits[name] = IKData(
            positions=z[f"{name}_positions"],
            joints=z[f"{name}_joints"],
            branch_labels=branch,
            metadata=meta,
        )
    return splits

"""High-level Neural IK model with normalization and device handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from neural_ik.data.normalization import Normalizer
from neural_ik.models.architectures import MLP


class NeuralIKModel:
    """End-to-end neural inverse kinematics predictor.

    Handles:
    - input/output normalization
    - device placement
    - batch and single-sample inference
    - checkpoint serialization
    """

    def __init__(
        self,
        network: MLP,
        normalizer: Normalizer | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.network = network
        self.normalizer = normalizer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.network.to(self.device)
        self.network.eval()

    def predict(
        self,
        positions: NDArray[np.floating[Any]] | list[float],
    ) -> NDArray[np.floating[Any]]:
        """Predict joint angles for one or more target positions.

        Parameters
        ----------
        positions : (2,) or (N, 2)

        Returns
        -------
        joints : (2,) or (N, 2)
        """
        pos = np.asarray(positions, dtype=np.float64)
        single = pos.ndim == 1
        if single:
            pos = pos[None, :]

        if self.normalizer is not None:
            pos_n = self.normalizer.transform_positions(pos)
        else:
            pos_n = pos

        with torch.no_grad():
            t = torch.from_numpy(pos_n.astype(np.float32)).to(self.device)
            out = self.network(t).cpu().numpy().astype(np.float64)

        if self.normalizer is not None:
            out = self.normalizer.inverse_joints(out)

        return out[0] if single else out

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": self.network.state_dict(),
            "arch": {
                "input_dim": self.network.net[0].in_features,
                "output_dim": self.network.net[-1].out_features,
                # Reconstruct hidden dims is non-trivial; store config externally
            },
        }
        if self.normalizer is not None:
            payload["normalizer"] = self.normalizer.state_dict()
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        network: MLP,
        device: str | torch.device | None = None,
    ) -> NeuralIKModel:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        network.load_state_dict(payload["state_dict"])
        normalizer = None
        if "normalizer" in payload:
            normalizer = Normalizer()
            normalizer.load_state_dict(payload["normalizer"])
        return cls(network=network, normalizer=normalizer, device=device)

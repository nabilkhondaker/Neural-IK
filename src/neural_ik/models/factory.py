"""Model factory from configuration dictionaries."""

from __future__ import annotations

from typing import Any, Sequence

from neural_ik.models.architectures import MLP


def build_model(cfg: dict[str, Any]) -> MLP:
    """Build an MLP from a config dict.

    Expected keys (with defaults):
        input_dim, output_dim, hidden_dims, activation, dropout
    """
    return MLP(
        input_dim=int(cfg.get("input_dim", 2)),
        output_dim=int(cfg.get("output_dim", 2)),
        hidden_dims=tuple(cfg.get("hidden_dims", [128, 128, 64])),
        activation=str(cfg.get("activation", "relu")),
        dropout=float(cfg.get("dropout", 0.0)),
    )


def model_size_presets() -> dict[str, Sequence[int]]:
    """Named width presets for scaling experiments."""
    return {
        "tiny": (32, 32),
        "small": (64, 64),
        "medium": (128, 128, 64),
        "large": (256, 256, 128, 64),
        "xlarge": (512, 256, 128, 64),
    }

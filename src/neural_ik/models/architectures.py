"""Configurable multi-layer perceptron for neural IK."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


class MLP(nn.Module):
    """Fully-connected network mapping (x, y) → (θ1, θ2).

    Parameters
    ----------
    input_dim : int
        Default 2 (Cartesian coordinates).
    output_dim : int
        Default 2 (joint angles).
    hidden_dims : sequence of int
        Width of each hidden layer.
    activation : str
        One of relu, gelu, tanh, silu, leaky_relu.
    dropout : float
        Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_dims: Sequence[int] = (128, 128, 64),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")
        act_cls = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

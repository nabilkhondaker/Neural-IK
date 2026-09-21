"""Loss functions for neural IK training."""

from __future__ import annotations

import torch
import torch.nn as nn


def joint_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error on joint angles (normalized space)."""
    return nn.functional.mse_loss(pred, target)


def position_fk_loss(
    pred_joints: torch.Tensor,
    target_pos: torch.Tensor,
    L1: float,
    L2: float,
) -> torch.Tensor:
    """Differentiable FK consistency loss.

    Computes end-effector position from predicted joints and measures
    Euclidean distance to the target position. Useful as an auxiliary
    loss that directly optimizes task-space accuracy.
    """
    th1 = pred_joints[..., 0]
    th2 = pred_joints[..., 1]
    x = L1 * torch.cos(th1) + L2 * torch.cos(th1 + th2)
    y = L1 * torch.sin(th1) + L2 * torch.sin(th1 + th2)
    pred_pos = torch.stack([x, y], dim=-1)
    return torch.mean(torch.sqrt(((pred_pos - target_pos) ** 2).sum(dim=-1) + 1e-12))

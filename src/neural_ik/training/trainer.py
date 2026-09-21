"""Training loop for neural IK models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_ik.models.losses import joint_mse_loss
from neural_ik.training.checkpointing import save_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    device: str | None = None
    checkpoint_dir: str = "artifacts/checkpoints"
    log_every: int = 10
    early_stopping_patience: int = 15


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1


class Trainer:
    """Simple supervised trainer for MLP inverse kinematics."""

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.cfg = cfg or TrainConfig()
        if self.cfg.device is None:
            self.cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.cfg.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or joint_mse_loss
        self.history = TrainHistory()

        if self.cfg.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
                momentum=0.9,
            )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> TrainHistory:
        patience_counter = 0
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            self.history.train_loss.append(train_loss)

            val_loss = float("nan")
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, train=False)
                self.history.val_loss.append(val_loss)

                if val_loss < self.history.best_val_loss:
                    self.history.best_val_loss = val_loss
                    self.history.best_epoch = epoch
                    patience_counter = 0
                    save_checkpoint(
                        self.model,
                        Path(self.cfg.checkpoint_dir) / "best.pt",
                        epoch=epoch,
                        val_loss=val_loss,
                        extra=extra_meta,
                    )
                else:
                    patience_counter += 1

            if epoch % self.cfg.log_every == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
                    epoch,
                    self.cfg.epochs,
                    train_loss,
                    val_loss,
                )

            if (
                val_loader is not None
                and self.cfg.early_stopping_patience > 0
                and patience_counter >= self.cfg.early_stopping_patience
            ):
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Final checkpoint
        save_checkpoint(
            self.model,
            Path(self.cfg.checkpoint_dir) / "last.pt",
            epoch=epoch,
            val_loss=self.history.val_loss[-1] if self.history.val_loss else None,
            extra=extra_meta,
        )
        return self.history

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total = 0.0
        n = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            if train:
                self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            if train:
                loss.backward()
                self.optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        return total / max(n, 1)

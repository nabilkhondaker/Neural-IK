"""Training pipeline integration."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from neural_ik.data.datasets import IKDataset
from neural_ik.data.generation import DatasetConfig, generate_ik_dataset
from neural_ik.data.normalization import Normalizer
from neural_ik.models.factory import build_model
from neural_ik.training.trainer import TrainConfig, Trainer


def test_short_training():
    splits = generate_ik_dataset(DatasetConfig(n_samples=1000, seed=0))
    norm = Normalizer().fit(splits["train"].positions, splits["train"].joints)
    train_ds = IKDataset(splits["train"].positions, splits["train"].joints, norm)
    val_ds = IKDataset(splits["val"].positions, splits["val"].joints, norm)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    vloader = DataLoader(val_ds, batch_size=128)
    model = build_model({"hidden_dims": [32, 16]})
    trainer = Trainer(model, TrainConfig(epochs=2, log_every=1, checkpoint_dir="/tmp/nik_ckpts"))
    hist = trainer.fit(loader, vloader)
    assert len(hist.train_loss) == 2

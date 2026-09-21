"""Neural prediction example (requires a trained checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from neural_ik.models.architectures import MLP
from neural_ik.models.neural_ik import NeuralIKModel

ckpt = Path("artifacts/experiments/baseline/neural_ik.pt")
if not ckpt.exists():
    print("No checkpoint found. Run scripts/train.py first.")
    sys.exit(0)

net = MLP(hidden_dims=(128, 128, 64))
model = NeuralIKModel.load(ckpt, network=net)
target = np.array([1.0, 0.4])
print("Predicted joints:", model.predict(target))

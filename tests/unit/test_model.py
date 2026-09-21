"""Model construction and inference tests."""

from __future__ import annotations

import numpy as np
import torch

from neural_ik.models.architectures import MLP
from neural_ik.models.factory import build_model
from neural_ik.models.neural_ik import NeuralIKModel


def test_mlp_forward():
    net = MLP(hidden_dims=(32, 16))
    x = torch.randn(8, 2)
    y = net(x)
    assert y.shape == (8, 2)


def test_build_model():
    net = build_model({"hidden_dims": [64, 32], "activation": "relu"})
    assert isinstance(net, MLP)


def test_neural_ik_predict():
    net = MLP(hidden_dims=(16, 16))
    model = NeuralIKModel(network=net, normalizer=None)
    out = model.predict(np.array([1.0, 0.5]))
    assert out.shape == (2,)
    batch = model.predict(np.random.randn(5, 2))
    assert batch.shape == (5, 2)

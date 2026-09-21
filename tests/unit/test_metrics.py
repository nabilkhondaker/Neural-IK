"""Metric tests."""

from __future__ import annotations

import numpy as np

from neural_ik.evaluation.metrics import position_error, summarize_errors


def test_position_error():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [1.0, 1.0]])
    e = position_error(a, b)
    np.testing.assert_allclose(e, [0.0, 1.0])


def test_summarize():
    errs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    s = summarize_errors(errs)
    assert s["n"] == 5
    assert abs(s["mae"] - 0.3) < 1e-9

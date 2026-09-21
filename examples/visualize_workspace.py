"""Plot reachable workspace."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_ik.robotics.models.planar_2r import Planar2R
from neural_ik.visualization.workspace import plot_workspace

robot = Planar2R()
rng = np.random.default_rng(0)
samples = robot.forward_kinematics(robot.sample_joints(2000, rng=rng))
ax = plot_workspace(robot, samples=samples)
out = Path("artifacts/plots/workspace.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
print("Wrote", out)

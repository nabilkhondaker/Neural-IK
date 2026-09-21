"""Analytical IK demo with both branches."""

from __future__ import annotations

import numpy as np

from neural_ik.robotics.models.planar_2r import Planar2R

robot = Planar2R()
targets = np.array([[1.0, 0.5], [0.5, 0.8], [1.5, 0.0]])
for t in targets:
    up, dn, reach = robot.both_branches(t)
    print(f"target={t} reachable={reach}")
    print(f"  up={up}  down={dn}")

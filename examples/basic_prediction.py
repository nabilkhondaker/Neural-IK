"""Basic analytical IK example."""

from __future__ import annotations

import numpy as np

from neural_ik.robotics.models.planar_2r import Planar2R

robot = Planar2R(link_1=1.0, link_2=0.75)
target = np.array([1.2, 0.4])
theta_up, ok = robot.analytical_ik(target, branch="elbow_up")
theta_dn, _ = robot.analytical_ik(target, branch="elbow_down")
print("Reachable:", ok)
print("Elbow-up:", theta_up)
print("Elbow-down:", theta_dn)
print("FK check:", robot.forward_kinematics(theta_up))

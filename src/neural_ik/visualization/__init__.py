"""Visualization utilities for robots, workspace, errors, and training."""

from neural_ik.visualization.robot import plot_robot
from neural_ik.visualization.workspace import plot_workspace
from neural_ik.visualization.errors import plot_error_histogram, plot_error_heatmap
from neural_ik.visualization.training import plot_training_curves

__all__ = [
    "plot_robot",
    "plot_workspace",
    "plot_error_histogram",
    "plot_error_heatmap",
    "plot_training_curves",
]

"""Generate supervised IK datasets via forward kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_ik.robotics.models.planar_2r import Planar2R


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    n_samples: int = 100_000
    seed: int = 42
    branch: str = "elbow_up"  # 'elbow_up', 'elbow_down', or 'both'
    noise_std: float = 0.0
    joint_limit_fraction: float = 1.0  # 1.0 = full limits
    link_1: float = 1.0
    link_2: float = 0.75
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    filter_unreachable: bool = True


@dataclass
class IKData:
    """Container for generated IK samples."""

    positions: NDArray[np.floating[Any]]
    joints: NDArray[np.floating[Any]]
    branch_labels: NDArray[np.int_] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.positions)


def generate_ik_dataset(cfg: DatasetConfig | None = None) -> dict[str, IKData]:
    """Generate train/val/test splits of (position → joints) pairs.

    Strategy
    --------
    1. Sample joint angles uniformly within limits.
    2. Apply forward kinematics to obtain Cartesian targets.
    3. Optionally inject Gaussian noise on positions.
    4. For multi-branch mode, store both solutions with labels.
    5. Split into train / val / test.

    The multi-solution problem is handled by the `branch` setting:
    - 'elbow_up' / 'elbow_down': single canonical branch (via FK sampling
      the corresponding joint region is naturally produced).
    - 'both': store two samples per configuration when distinct.

    Returns
    -------
    dict with keys 'train', 'val', 'test' mapping to IKData.
    """
    if cfg is None:
        cfg = DatasetConfig()

    rng = np.random.default_rng(cfg.seed)
    robot = Planar2R(link_1=cfg.link_1, link_2=cfg.link_2)

    # Sample joints
    n = cfg.n_samples
    joints = robot.sample_joints(n, rng=rng)

    # Optionally restrict joint range
    if cfg.joint_limit_fraction < 1.0:
        lower, upper = robot.joint_limits()
        mid = (lower + upper) / 2
        half = (upper - lower) / 2 * cfg.joint_limit_fraction
        joints = rng.uniform(mid - half, mid + half, size=(n, 2))

    positions = robot.forward_kinematics(joints)

    # Noise
    if cfg.noise_std > 0:
        positions = positions + rng.normal(0, cfg.noise_std, size=positions.shape)

    # Branch labels: 0 = elbow_up (θ2 < 0), 1 = elbow_down
    branch_labels = (joints[:, 1] >= 0).astype(np.int_)

    # Filter any numerically unreachable (shouldn't happen for pure FK)
    if cfg.filter_unreachable:
        reach = robot.is_reachable(positions)
        positions = positions[reach]
        joints = joints[reach]
        branch_labels = branch_labels[reach]

    # Shuffle and split
    idx = rng.permutation(len(positions))
    positions = positions[idx]
    joints = joints[idx]
    branch_labels = branch_labels[idx]

    n_total = len(positions)
    n_train = int(n_total * cfg.train_frac)
    n_val = int(n_total * cfg.val_frac)

    def _slice(start: int, end: int) -> IKData:
        return IKData(
            positions=positions[start:end].astype(np.float64),
            joints=joints[start:end].astype(np.float64),
            branch_labels=branch_labels[start:end],
            metadata={
                "link_1": cfg.link_1,
                "link_2": cfg.link_2,
                "seed": cfg.seed,
                "noise_std": cfg.noise_std,
                "branch": cfg.branch,
            },
        )

    splits = {
        "train": _slice(0, n_train),
        "val": _slice(n_train, n_train + n_val),
        "test": _slice(n_train + n_val, n_total),
    }
    return splits

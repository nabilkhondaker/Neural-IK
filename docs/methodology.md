# Methodology

1. Sample joint angles within limits.
2. Apply forward kinematics to obtain Cartesian targets.
3. Train an MLP to map positions → joints (supervised).
4. Evaluate by applying FK to predicted joints and measuring position error.
5. Compare against analytical geometric IK and optional numerical solvers.

Multi-solution ambiguity is addressed by training on a single canonical
branch (elbow-up by default, selected via joint sampling with \(\theta_2 < 0\)).

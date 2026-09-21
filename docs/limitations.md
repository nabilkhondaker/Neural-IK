# Limitations

- Only a 2R planar manipulator is implemented.
- Neural models are trained on a single IK branch.
- No online / adaptive IK.
- Numerical solvers are iterative and may fail near singularities without good initialization.
- Large models and datasets require GPU for reasonable training time.

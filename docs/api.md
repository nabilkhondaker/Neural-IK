# API Overview

```python
from neural_ik.robotics.models.planar_2r import Planar2R
from neural_ik.data.generation import generate_ik_dataset, DatasetConfig
from neural_ik.models.factory import build_model
from neural_ik.models.neural_ik import NeuralIKModel
from neural_ik.evaluation.evaluator import evaluate_solver

robot = Planar2R()
theta, ok = robot.analytical_ik([1.0, 0.5])
```

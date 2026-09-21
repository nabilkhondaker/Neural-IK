# Architecture

The package is organized into subsystems:

| Package | Role |
|---------|------|
| `robotics` | Robot models, FK/IK, workspace, singularities |
| `data` | Dataset generation, normalization, validation |
| `models` | MLP architectures and NeuralIKModel wrapper |
| `training` | Trainer, checkpoints, seeds |
| `evaluation` | Metrics, benchmarks, robustness |
| `experiments` | Runnable experiment drivers |
| `visualization` | Robot, workspace, error, training plots |
| `reporting` | Text report generation |
| `cli` | Click-based command line interface |
| `utils` | Config, logging, paths |

Data flow for the baseline experiment:

```
Config → Dataset generation (FK) → Normalize → Train MLP
                                              ↓
Test targets → Neural predict / Analytical IK → FK → Position error
```

# Contributing

Thank you for your interest in Neural IK Lab.

## Development setup

```bash
pip install -e ".[dev]"
pre-commit install  # optional
```

## Guidelines

- Prefer type hints and docstrings on public APIs.
- Add unit tests for new kinematics or metric logic.
- Do not commit large datasets or model weights.
- Run `ruff check` and `pytest` before opening a PR.

## Pull requests

Use the PR template. Describe the motivation, the change, and how it was tested.

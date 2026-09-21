"""Path helpers."""

from __future__ import annotations

from pathlib import Path

# Repository root relative to this file: src/neural_ik/utils/paths.py → ../../../
REPO_ROOT = Path(__file__).resolve().parents[3]


def artifact_dir(name: str = "") -> Path:
    d = REPO_ROOT / "artifacts" / name if name else REPO_ROOT / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d

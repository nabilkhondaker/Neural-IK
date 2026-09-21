#!/usr/bin/env python3
"""Generate research report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_ik.reporting.generator import generate_text_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/reports/research_report.md")
    args = parser.parse_args()
    text = generate_text_report(output_path=args.output)
    print(f"Wrote {args.output} ({len(text)} chars)")


if __name__ == "__main__":
    main()

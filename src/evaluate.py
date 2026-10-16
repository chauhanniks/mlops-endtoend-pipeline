"""Enforce a release quality gate against the saved candidate metrics."""
import argparse
import json
import os
from pathlib import Path


def evaluate(metrics_path: str, min_f1: float) -> dict:
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    score = float(metrics["f1_score"])
    print(f"Candidate F1={score:.4f}; required minimum={min_f1:.4f}")
    if score < min_f1:
        raise SystemExit("QUALITY GATE FAILED: candidate model rejected.")
    print("QUALITY GATE PASSED")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts/metrics.json")
    parser.add_argument(
        "--min-f1",
        type=float,
        default=float(os.getenv("MIN_F1", "0.90")),
    )
    args = parser.parse_args()
    evaluate(args.metrics, args.min_f1)

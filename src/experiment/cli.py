"""Command-line entry point for repeatable real-video experiments."""
from __future__ import annotations

import argparse
import json

from src.experiment.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run human-video → robot-scene planning experiment."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    report = run_experiment(args.config)
    print(json.dumps({
        "experiment": report["experiment"],
        "selected_id": report["decision"]["selected_id"],
        "selected_source": report["decision"]["selected_source"],
        "human_demo_selected": report["decision"]["human_demo_selected"],
        "report": report["artifacts"]["report"],
        "overlay": report["artifacts"]["overlay"],
    }, indent=2))


if __name__ == "__main__":
    main()

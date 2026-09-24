"""Demo: an unseen obstacle appears after planning; vision triggers recovery."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from simulation.tabletop import TabletopSimulator
from src.agent.closed_loop import run_closed_loop


def main() -> None:
    simulator = TabletopSimulator(
        effector_xy=(0.10, 0.82),
        target_xy=(0.88, 0.18),
    )
    surprise = [(0.43, 0.43, 0.60, 0.64)]
    simulator.schedule_obstacles(attempt=1, obstacles=surprise)

    before = simulator.render()
    result = run_closed_loop(simulator, max_attempts=3)
    after = simulator.render()

    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "closed_loop_before.png"), before)
    cv2.imwrite(str(output_dir / "closed_loop_after.png"), after)

    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit("Closed-loop recovery demo did not reach the visual goal")


if __name__ == "__main__":
    main()

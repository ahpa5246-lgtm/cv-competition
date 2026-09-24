"""Synthetic benchmark for the value of visual re-observation and replanning.

This is an orchestration benchmark, not a real-robot result. Each episode hides
an obstacle during planning and reveals it immediately before execution.
"""
from __future__ import annotations

import argparse
import json
import random

from simulation.tabletop import TabletopSimulator
from src.agent.closed_loop import run_closed_loop
from src.pipeline import plan_from_robot_state
from src.core.types import RobotTrajectory


def _trajectory_from_plan(plan: dict) -> RobotTrajectory:
    data = plan["selected_trajectory"]
    return RobotTrajectory(
        trajectory_id=data["trajectory_id"],
        waypoints=[tuple(point) for point in data["waypoints"]],
        source=data["source"],
        metadata=dict(data["metadata"]),
    )


def _scenario(rng: random.Random):
    start = (rng.uniform(0.07, 0.14), rng.uniform(0.75, 0.88))
    target = (rng.uniform(0.84, 0.93), rng.uniform(0.12, 0.26))
    t = rng.uniform(0.46, 0.56)
    cx = start[0] + (target[0] - start[0]) * t
    cy = start[1] + (target[1] - start[1]) * t
    half_w = rng.uniform(0.06, 0.085)
    half_h = rng.uniform(0.07, 0.10)
    obstacle = (
        max(0.0, cx - half_w),
        max(0.0, cy - half_h),
        min(1.0, cx + half_w),
        min(1.0, cy + half_h),
    )
    return start, target, obstacle


def benchmark(episodes: int = 50, seed: int = 42) -> dict[str, float | int]:
    rng = random.Random(seed)
    open_loop_success = 0
    open_loop_collisions = 0
    closed_loop_success = 0
    closed_loop_attempts = 0

    for _ in range(episodes):
        start, target, obstacle = _scenario(rng)

        open_sim = TabletopSimulator(effector_xy=start, target_xy=target)
        open_sim.schedule_obstacles(1, [obstacle])
        plan = plan_from_robot_state(start, target, num_candidates=9)
        execution = open_sim.execute(_trajectory_from_plan(plan))
        open_loop_success += int(execution.reached_target)
        open_loop_collisions += int(execution.collision)

        closed_sim = TabletopSimulator(effector_xy=start, target_xy=target)
        closed_sim.schedule_obstacles(1, [obstacle])
        result = run_closed_loop(closed_sim, max_attempts=3)
        closed_loop_success += int(result["success"])
        closed_loop_attempts += int(result["attempts"])

    return {
        "episodes": episodes,
        "seed": seed,
        "open_loop_success_rate": open_loop_success / episodes,
        "open_loop_collision_rate": open_loop_collisions / episodes,
        "closed_loop_success_rate": closed_loop_success / episodes,
        "closed_loop_mean_attempts": closed_loop_attempts / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = benchmark(args.episodes, args.seed)
    print(json.dumps(result, indent=2))

    if result["closed_loop_success_rate"] <= result["open_loop_success_rate"]:
        raise SystemExit("Closed loop did not improve success in the synthetic recovery benchmark")


if __name__ == "__main__":
    main()

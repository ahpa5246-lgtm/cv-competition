"""Synthetic ablation: direct execution versus imagine-before-act planning.

This benchmark validates pipeline logic only. It is not evidence of real-robot
performance and must not be reported as such.
"""
from __future__ import annotations

import argparse
import json
import random

from src.prediction.future_model import HeuristicWorldModel
from src.robot.trajectory import generate_trajectory


def run_episode(rng: random.Random) -> dict[str, bool]:
    start = (0.08, rng.uniform(0.68, 0.90))
    goal = (0.92, rng.uniform(0.10, 0.32))

    mid_x = rng.uniform(0.43, 0.57)
    mid_y = (start[1] + goal[1]) / 2 + rng.uniform(-0.06, 0.06)
    half_w, half_h = rng.uniform(0.06, 0.10), rng.uniform(0.07, 0.12)
    obstacle = [(mid_x-half_w, mid_y-half_h, mid_x+half_w, mid_y+half_h)]

    model = HeuristicWorldModel(success_radius=0.08)

    direct = generate_trajectory(start, goal, num_candidates=1)[0]
    direct_outcome = model.predict(direct, goal, obstacle)

    candidates = generate_trajectory(
        start,
        goal,
        num_candidates=7,
        max_lateral_offset=0.45,
    )
    outcomes = [model.predict(candidate, goal, obstacle) for candidate in candidates]
    planned = max(
        outcomes,
        key=lambda x: 2*x.success_probability - 3*x.collision_risk - x.target_distance,
    )

    return {
        "direct_success": direct_outcome.collision_risk == 0 and direct_outcome.success_probability > 0.5,
        "direct_collision": direct_outcome.collision_risk > 0,
        "planned_success": planned.collision_risk == 0 and planned.success_probability > 0.5,
        "planned_collision": planned.collision_risk > 0,
    }


def benchmark(episodes: int = 200, seed: int = 42) -> dict[str, float | int]:
    rng = random.Random(seed)
    rows = [run_episode(rng) for _ in range(episodes)]

    def rate(key: str) -> float:
        return sum(int(row[key]) for row in rows) / episodes

    return {
        "episodes": episodes,
        "seed": seed,
        "direct_success_rate": rate("direct_success"),
        "direct_collision_rate": rate("direct_collision"),
        "planned_success_rate": rate("planned_success"),
        "planned_collision_rate": rate("planned_collision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.episodes, args.seed), indent=2))


if __name__ == "__main__":
    main()

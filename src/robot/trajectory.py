"""Generate alternative robot trajectories around a task intent."""
from __future__ import annotations

from src.core.types import Point2D, RobotTrajectory


def _lerp(a: Point2D, b: Point2D, t: float) -> Point2D:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def generate_trajectory(
    start: Point2D,
    goal: Point2D,
    *,
    num_candidates: int = 5,
    num_waypoints: int = 25,
    max_lateral_offset: float = 0.25,
) -> list[RobotTrajectory]:
    if num_candidates < 1 or num_waypoints < 2:
        raise ValueError("num_candidates >= 1 and num_waypoints >= 2 are required")

    dx, dy = goal[0] - start[0], goal[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1e-8)
    normal = (-dy / length, dx / length)

    if num_candidates == 1:
        offsets = [0.0]
    else:
        offsets = [
            -max_lateral_offset + (2 * max_lateral_offset * i / (num_candidates - 1))
            for i in range(num_candidates)
        ]
        offsets[num_candidates // 2] = 0.0

    candidates: list[RobotTrajectory] = []
    for i, offset in enumerate(offsets):
        waypoints: list[Point2D] = []
        for step in range(num_waypoints):
            t = step / (num_waypoints - 1)
            x, y = _lerp(start, goal, t)
            arch = 4.0 * t * (1.0 - t)
            waypoints.append((x + normal[0] * offset * arch, y + normal[1] * offset * arch))
        candidates.append(
            RobotTrajectory(
                trajectory_id=f"candidate-{i:02d}",
                waypoints=waypoints,
                source="generated-alternative",
                metadata={"lateral_offset": offset},
            )
        )
    return candidates

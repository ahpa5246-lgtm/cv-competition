"""Retarget task-centric human motion into a robot workspace."""
from __future__ import annotations

from src.core.types import MotionIntent, Point2D
from src.motion.normalization import denormalize_point


def map_to_robot(
    motion: MotionIntent,
    robot_config: dict,
) -> tuple[Point2D, Point2D, list[Point2D]]:
    workspace = tuple(robot_config.get("workspace", (0.0, 0.0, 1.0, 1.0)))
    if len(workspace) != 4:
        raise ValueError("robot workspace must be [x_min, y_min, x_max, y_max]")

    start = denormalize_point(motion.start_xy, workspace)
    goal = denormalize_point(motion.goal_xy, workspace)
    path = [denormalize_point(point, workspace) for point in motion.path_xy]
    return start, goal, path

"""Task-centric representation of an observed human demonstration."""
from __future__ import annotations

from src.core.types import MotionIntent, Point2D
from src.motion.normalization import normalize_motion, normalize_point


def encode_motion(
    observed_path: list[Point2D],
    target_xy: Point2D,
    frame_size: tuple[int, int],
    *,
    task: str = "reach_target",
) -> MotionIntent:
    if len(observed_path) < 2:
        raise ValueError("At least two observed points are required to encode motion")

    normalized_path = normalize_motion(observed_path, frame_size)
    target = normalize_point(target_xy, frame_size)

    return MotionIntent(
        task=task,
        start_xy=normalized_path[0],
        goal_xy=target,
        path_xy=normalized_path,
        coordinate_space="normalized",
    )

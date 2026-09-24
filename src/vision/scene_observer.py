"""Build a normalized scene state from a rendered/camera frame using OpenCV."""
from __future__ import annotations

import cv2
import numpy as np

from src.core.types import BBox, Point2D, SceneState
from src.vision.object_tracking import find_colored_regions, largest_region_centroid


def _normalize_point(point: Point2D | None, width: int, height: int) -> Point2D | None:
    if point is None:
        return None
    return point[0] / width, point[1] / height


def _normalize_box(box: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    return x1 / width, y1 / height, x2 / width, y2 / height


def observe_scene(
    frame: np.ndarray,
    *,
    effector_hsv: tuple[tuple[int, int, int], tuple[int, int, int]],
    target_hsv: tuple[tuple[int, int, int], tuple[int, int, int]],
    obstacle_hsv: tuple[tuple[int, int, int], tuple[int, int, int]],
    frame_index: int = 0,
    timestamp_s: float = 0.0,
) -> SceneState:
    height, width = frame.shape[:2]
    effector = largest_region_centroid(frame, *effector_hsv)
    target = largest_region_centroid(frame, *target_hsv)
    boxes = find_colored_regions(frame, *obstacle_hsv, min_area=100.0)

    confidence = 1.0
    if effector is None:
        confidence -= 0.45
    if target is None:
        confidence -= 0.45

    return SceneState(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        hand_xy=_normalize_point(effector, width, height),
        target_xy=_normalize_point(target, width, height),
        obstacle_boxes=[_normalize_box(box, width, height) for box in boxes],
        confidence=max(0.0, confidence),
    )

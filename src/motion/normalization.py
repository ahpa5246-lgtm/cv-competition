"""Coordinate normalization for cross-embodiment transfer."""
from __future__ import annotations

from src.core.types import Point2D


def normalize_point(point: Point2D, frame_size: tuple[int, int]) -> Point2D:
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    return point[0] / width, point[1] / height


def denormalize_point(point: Point2D, workspace: tuple[float, float, float, float]) -> Point2D:
    x_min, y_min, x_max, y_max = workspace
    return (
        x_min + point[0] * (x_max - x_min),
        y_min + point[1] * (y_max - y_min),
    )


def normalize_motion(points: list[Point2D], frame_size: tuple[int, int]) -> list[Point2D]:
    return [normalize_point(point, frame_size) for point in points]

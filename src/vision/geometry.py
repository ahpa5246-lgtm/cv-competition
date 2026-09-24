"""Camera/tabletop geometry helpers implemented with OpenCV."""
from __future__ import annotations

import cv2
import numpy as np

from src.core.types import Point2D


def estimate_homography(
    image_points: list[Point2D],
    world_points: list[Point2D],
) -> np.ndarray:
    if len(image_points) < 4 or len(image_points) != len(world_points):
        raise ValueError("Need at least four matched image/world points")

    src = np.asarray(image_points, dtype=np.float32)
    dst = np.asarray(world_points, dtype=np.float32)
    matrix, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
    if matrix is None or mask is None:
        raise RuntimeError("OpenCV could not estimate a valid homography")
    return matrix


def transform_points(points: list[Point2D], homography: np.ndarray) -> list[Point2D]:
    if not points:
        return []
    src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, homography).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in dst]

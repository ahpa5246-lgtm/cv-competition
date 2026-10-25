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
    if src.shape != (len(image_points), 2) or dst.shape != src.shape or not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Calibration points must be finite coordinate pairs")
    matrix, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
    if matrix is None or mask is None:
        raise RuntimeError("OpenCV could not estimate a valid homography")
    return matrix


def transform_points(points: list[Point2D], homography: np.ndarray) -> list[Point2D]:
    if not points:
        return []
    homography = np.asarray(homography, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("Homography must be a finite 3 by 3 matrix")
    src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if not np.isfinite(src).all():
        raise ValueError("Points must be finite")
    dst = cv2.perspectiveTransform(src, homography).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in dst]

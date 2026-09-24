"""Lightweight OpenCV scene tracking used by the MVP baseline.

The competition system can later replace HSV segmentation with OpenCV DNN
detectors/segmenters without changing downstream planning interfaces.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.core.types import BBox, Point2D


def _mask_for_range(
    frame: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray(lower_hsv, dtype=np.uint8),
        np.asarray(upper_hsv, dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def find_colored_regions(
    frame: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
    *,
    min_area: float = 40.0,
) -> list[BBox]:
    mask = _mask_for_range(frame, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[BBox] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return boxes


def largest_region_centroid(
    frame: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
    *,
    min_area: float = 40.0,
) -> Point2D | None:
    mask = _mask_for_range(frame, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def track_colored_object(
    frames: list[np.ndarray],
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
) -> list[Point2D | None]:
    return [largest_region_centroid(frame, lower_hsv, upper_hsv) for frame in frames]

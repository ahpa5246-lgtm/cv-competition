"""OpenCV visualization for an experiment decision trace."""
from __future__ import annotations

import cv2
import numpy as np

from src.core.types import BBox, Point2D


def _px(point: Point2D, width: int, height: int) -> tuple[int, int]:
    return int(round(point[0] * width)), int(round(point[1] * height))


def _draw_path(
    frame: np.ndarray,
    points: list[Point2D],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    height, width = frame.shape[:2]
    polyline = np.asarray([_px(point, width, height) for point in points], dtype=np.int32)
    cv2.polylines(frame, [polyline], False, color, thickness, cv2.LINE_AA)


def draw_experiment_overlay(
    frame: np.ndarray,
    *,
    start_xy: Point2D,
    target_xy: Point2D,
    obstacles: list[BBox],
    human_path: list[Point2D],
    selected_path: list[Point2D],
    selected_source: str,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    for box in obstacles:
        x1, y1, x2, y2 = box
        cv2.rectangle(
            canvas,
            _px((x1, y1), width, height),
            _px((x2, y2), width, height),
            (255, 180, 0),
            2,
        )

    # Proposed human transfer is thin magenta; the selected executable path is
    # thick black so screenshots remain legible even when printed.
    _draw_path(canvas, human_path, (200, 0, 200), 2)
    _draw_path(canvas, selected_path, (20, 20, 20), 4)

    cv2.circle(canvas, _px(start_xy, width, height), 8, (0, 180, 0), -1)
    cv2.circle(canvas, _px(target_xy, width, height), 10, (0, 0, 220), 2)
    cv2.putText(
        canvas,
        f"selected: {selected_source}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "magenta=human transfer | black=selected",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas

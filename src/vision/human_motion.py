"""Extract temporal motion cues with OpenCV optical flow."""
from __future__ import annotations

import cv2
import numpy as np


def extract_motion(
    frames: list[np.ndarray],
    *,
    roi: tuple[int, int, int, int] | None = None,
    magnitude_threshold: float = 0.8,
) -> list[tuple[float, float, float]]:
    if len(frames) < 2:
        return []

    def crop(frame: np.ndarray) -> np.ndarray:
        if roi is None:
            return frame
        x1, y1, x2, y2 = roi
        return frame[y1:y2, x1:x2]

    previous = cv2.cvtColor(crop(frames[0]), cv2.COLOR_BGR2GRAY)
    observations: list[tuple[float, float, float]] = []

    for frame in frames[1:]:
        current = cv2.cvtColor(crop(frame), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        active = magnitude >= magnitude_threshold

        if np.any(active):
            dx = float(np.median(flow[..., 0][active]))
            dy = float(np.median(flow[..., 1][active]))
            mag = float(np.median(magnitude[active]))
        else:
            dx = dy = mag = 0.0

        observations.append((dx, dy, mag))
        previous = current

    return observations

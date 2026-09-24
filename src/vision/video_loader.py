"""Video loading utilities built on OpenCV 5."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class VideoSequence:
    frames: list[np.ndarray]
    fps: float
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps if self.fps > 0 else 0.0


def load_video(path: str, *, max_frames: int | None = None, stride: int = 1) -> VideoSequence:
    if stride < 1:
        raise ValueError("stride must be >= 1")

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames were decoded from: {path}")

    return VideoSequence(frames=frames, fps=fps, width=width, height=height)

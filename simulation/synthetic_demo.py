"""Reproducible end-to-end demo using generated frames and OpenCV perception."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.motion.normalization import normalize_point
from src.pipeline import plan_from_observation
from src.vision.object_tracking import track_colored_object

WIDTH, HEIGHT = 640, 420
HAND_HSV = ((45, 120, 80), (85, 255, 255))      # green
TARGET_HSV = ((0, 140, 100), (10, 255, 255))     # red


def make_frames(count: int = 50) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    start = np.array([90.0, 330.0])
    end = np.array([500.0, 115.0])

    for i in range(count):
        t = i / (count - 1)
        hand = (1 - t) * start + t * end
        frame = np.full((HEIGHT, WIDTH, 3), 245, dtype=np.uint8)
        cv2.rectangle(frame, (265, 175), (365, 300), (255, 0, 0), -1)
        cv2.circle(frame, (520, 105), 28, (0, 0, 255), -1)
        cv2.circle(frame, tuple(hand.astype(int)), 18, (0, 255, 0), -1)
        frames.append(frame)
    return frames


def draw_plan(frame: np.ndarray, selected: list[tuple[float, float]]) -> np.ndarray:
    canvas = frame.copy()
    pixels = np.asarray(
        [[int(x * WIDTH), int(y * HEIGHT)] for x, y in selected],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [pixels], False, (0, 0, 0), 4, cv2.LINE_AA)
    return canvas


def main() -> None:
    frames = make_frames()
    hand_track = [
        p for p in track_colored_object(frames, *HAND_HSV)
        if p is not None
    ]
    targets = [
        p for p in track_colored_object(frames, *TARGET_HSV)
        if p is not None
    ]
    if len(hand_track) < 2 or not targets:
        raise RuntimeError("Synthetic perception failed")

    obstacle_pixels = (265.0, 175.0, 365.0, 300.0)
    p1 = normalize_point((obstacle_pixels[0], obstacle_pixels[1]), (WIDTH, HEIGHT))
    p2 = normalize_point((obstacle_pixels[2], obstacle_pixels[3]), (WIDTH, HEIGHT))
    obstacles = [(p1[0], p1[1], p2[0], p2[1])]

    result = plan_from_observation(
        hand_track,
        targets[-1],
        (WIDTH, HEIGHT),
        obstacles=obstacles,
        robot_config={
            "workspace": (0.0, 0.0, 1.0, 1.0),
            "max_lateral_offset": 0.42,
        },
        num_candidates=7,
    )

    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [tuple(p) for p in result["selected_trajectory"]["waypoints"]]
    visualization = draw_plan(frames[-1], selected)
    cv2.imwrite(str(output_dir / "synthetic_plan.png"), visualization)

    print(json.dumps({
        "selected": result["selected_trajectory"]["trajectory_id"],
        "verification": result["verification"],
        "ranking": result["ranking"],
        "visualization": str(output_dir / "synthetic_plan.png"),
    }, indent=2))


if __name__ == "__main__":
    main()

"""CI/demo for the YAML-driven experiment harness."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from simulation.skill_transfer_demo import write_human_demo
from src.experiment.runner import run_experiment


def main() -> None:
    root = Path("outputs/experiments/harness-demo-inputs")
    root.mkdir(parents=True, exist_ok=True)

    video_path = root / "human_demo.avi"
    scene_path = root / "robot_scene.png"
    config_path = root / "experiment.yaml"

    write_human_demo(video_path)

    frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.circle(frame, (95, 405), 12, (0, 255, 0), -1)
    cv2.circle(frame, (550, 80), 24, (0, 0, 255), -1)
    cv2.rectangle(frame, (285, 220), (365, 315), (255, 0, 0), -1)
    if not cv2.imwrite(str(scene_path), frame):
        raise RuntimeError("Could not write harness robot-scene image")

    config = {
        "name": "ci-real-video-harness",
        "task": "reach_target",
        "inputs": {
            "demonstration_video": str(video_path.resolve()),
            "robot_scene_image": str(scene_path.resolve()),
            "max_frames": 100,
        },
        "vision": {
            "human_hand_hsv": {"lower": [45, 120, 80], "upper": [85, 255, 255]},
            "robot_effector_hsv": {"lower": [45, 120, 80], "upper": [85, 255, 255]},
            "target_hsv": {"lower": [0, 140, 100], "upper": [10, 255, 255]},
            "obstacle_hsv": {"lower": [100, 120, 80], "upper": [140, 255, 255]},
        },
        "planning": {
            "num_candidates": 9,
            "max_lateral_offset": 0.45,
        },
        "output": {"directory": str(Path("outputs/experiments").resolve())},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = run_experiment(config_path)
    summary = {
        "tracked_points": report["demonstration_perception"]["tracked_points"],
        "scene_obstacles": len(report["robot_scene"]["obstacle_boxes"]),
        "selected_id": report["decision"]["selected_id"],
        "selected_source": report["decision"]["selected_source"],
        "report": report["artifacts"]["report"],
        "overlay": report["artifacts"]["overlay"],
    }
    print(json.dumps(summary, indent=2))

    if summary["tracked_points"] < 50:
        raise SystemExit("Experiment harness lost too much demonstration motion")
    if summary["scene_obstacles"] != 1:
        raise SystemExit("Experiment harness did not observe the robot-scene obstacle")
    if not Path(summary["report"]).exists() or not Path(summary["overlay"]).exists():
        raise SystemExit("Experiment harness did not persist evidence artifacts")


if __name__ == "__main__":
    main()

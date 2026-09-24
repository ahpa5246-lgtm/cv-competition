from pathlib import Path

import cv2
import numpy as np
import yaml

from simulation.skill_transfer_demo import write_human_demo
from src.experiment.runner import run_experiment


def test_yaml_experiment_writes_report_and_overlay(tmp_path: Path):
    video_path = tmp_path / "human_demo.avi"
    scene_path = tmp_path / "robot_scene.png"
    output_dir = tmp_path / "results"
    config_path = tmp_path / "experiment.yaml"

    write_human_demo(video_path)

    scene = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.circle(scene, (90, 410), 12, (0, 255, 0), -1)
    cv2.circle(scene, (550, 80), 24, (0, 0, 255), -1)
    assert cv2.imwrite(str(scene_path), scene)

    config = {
        "name": "pytest-real-video-harness",
        "task": "reach_target",
        "inputs": {
            "demonstration_video": str(video_path),
            "robot_scene_image": str(scene_path),
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
        "output": {"directory": str(output_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = run_experiment(config_path)

    assert report["demonstration_perception"]["tracked_points"] == 60
    assert report["decision"]["selected_id"] == "human-demo"
    assert report["decision"]["selected_source"] == "human-demonstration-transfer"
    assert Path(report["artifacts"]["report"]).exists()
    assert Path(report["artifacts"]["overlay"]).exists()

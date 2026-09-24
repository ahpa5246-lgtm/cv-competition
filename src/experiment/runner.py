"""Run one reproducible video-to-robot-scene experiment from YAML."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import cv2
import yaml

from src.core.types import RobotTrajectory
from src.experiment.visualization import draw_experiment_overlay
from src.pipeline import learn_motion_intent_from_video, plan_from_robot_state
from src.robot.skill_transfer import trajectory_from_demonstration
from src.vision.scene_observer import observe_scene


def _hsv_range(value: Any, name: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if not isinstance(value, dict) or "lower" not in value or "upper" not in value:
        raise ValueError(f"{name} must contain lower and upper HSV triplets")
    lower = tuple(int(x) for x in value["lower"])
    upper = tuple(int(x) for x in value["upper"])
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError(f"{name} HSV values must contain exactly 3 integers")
    return lower, upper


def _resolve(path_value: str, config_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    # Prefer paths relative to the config file. If that does not exist, fall
    # back to the repository/current working directory for convenient examples.
    relative = (config_path.parent / path).resolve()
    if relative.exists():
        return relative
    return path.resolve()


def load_experiment_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    return config, config_path


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    config, resolved_config_path = load_experiment_config(config_path)

    experiment_name = str(config.get("name", "video-to-robot-experiment"))
    inputs = config.get("inputs", {})
    vision = config.get("vision", {})
    planning = config.get("planning", {})
    output = config.get("output", {})

    demonstration_video = _resolve(str(inputs["demonstration_video"]), resolved_config_path)
    robot_scene_image = _resolve(str(inputs["robot_scene_image"]), resolved_config_path)

    hand_hsv = _hsv_range(vision["human_hand_hsv"], "human_hand_hsv")
    target_hsv = _hsv_range(vision["target_hsv"], "target_hsv")
    effector_hsv = _hsv_range(vision["robot_effector_hsv"], "robot_effector_hsv")
    obstacle_hsv = _hsv_range(vision["obstacle_hsv"], "obstacle_hsv")

    intent, demo_perception = learn_motion_intent_from_video(
        str(demonstration_video),
        hand_hsv=hand_hsv,
        target_hsv=target_hsv,
        max_frames=inputs.get("max_frames"),
        task=str(config.get("task", "reach_target")),
    )

    scene_frame = cv2.imread(str(robot_scene_image))
    if scene_frame is None:
        raise FileNotFoundError(f"OpenCV could not read robot scene: {robot_scene_image}")

    scene = observe_scene(
        scene_frame,
        effector_hsv=effector_hsv,
        target_hsv=target_hsv,
        obstacle_hsv=obstacle_hsv,
    )
    if scene.hand_xy is None:
        raise RuntimeError("Robot effector was not detected in the scene image")
    if scene.target_xy is None:
        raise RuntimeError("Target was not detected in the scene image")

    num_candidates = int(planning.get("num_candidates", 9))
    max_lateral_offset = float(planning.get("max_lateral_offset", 0.45))
    plan = plan_from_robot_state(
        scene.hand_xy,
        scene.target_xy,
        obstacles=scene.obstacle_boxes,
        robot_config={
            "workspace": (0.0, 0.0, 1.0, 1.0),
            "max_lateral_offset": max_lateral_offset,
        },
        num_candidates=num_candidates,
        demonstration_intent=intent,
    )

    human_candidate = trajectory_from_demonstration(
        intent,
        scene.hand_xy,
        scene.target_xy,
    )
    selected_data = plan["selected_trajectory"]
    selected = RobotTrajectory(
        trajectory_id=selected_data["trajectory_id"],
        waypoints=[tuple(point) for point in selected_data["waypoints"]],
        source=selected_data["source"],
        metadata=dict(selected_data["metadata"]),
    )

    output_dir_value = str(output.get("directory", "outputs/experiments"))
    output_root = Path(output_dir_value).expanduser()
    if not output_root.is_absolute():
        output_root = output_root.resolve()
    run_dir = output_root / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    overlay = draw_experiment_overlay(
        scene_frame,
        start_xy=scene.hand_xy,
        target_xy=scene.target_xy,
        obstacles=scene.obstacle_boxes,
        human_path=human_candidate.waypoints,
        selected_path=selected.waypoints,
        selected_source=selected.source,
    )
    overlay_path = run_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": experiment_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
        },
        "inputs": {
            "demonstration_video": str(demonstration_video),
            "robot_scene_image": str(robot_scene_image),
            "config": str(resolved_config_path),
        },
        "demonstration_perception": demo_perception,
        "learned_intent": asdict(intent),
        "robot_scene": asdict(scene),
        "human_transfer": asdict(human_candidate),
        "planning": plan,
        "decision": {
            "selected_id": selected.trajectory_id,
            "selected_source": selected.source,
            "human_demo_selected": selected.trajectory_id == "human-demo",
        },
        "artifacts": {
            "overlay": str(overlay_path),
        },
    }

    report_path = run_dir / "report.json"
    report["artifacts"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

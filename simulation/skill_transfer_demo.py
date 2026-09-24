"""End-to-end: human video → transferred skill → safe adaptation."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from simulation.tabletop import TabletopSimulator
from src.agent.closed_loop import (
    DEFAULT_EFFECTOR_HSV,
    DEFAULT_TARGET_HSV,
    run_closed_loop,
)
from src.pipeline import learn_motion_intent_from_video, plan_from_robot_state
from src.robot.skill_transfer import trajectory_from_demonstration

WIDTH, HEIGHT = 640, 480


def _bezier(t: float, p0, p1, p2):
    one = 1.0 - t
    return (
        one * one * p0[0] + 2 * one * t * p1[0] + t * t * p2[0],
        one * one * p0[1] + 2 * one * t * p1[1] + t * t * p2[1],
    )


def write_human_demo(path: Path, frame_count: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter could not create the synthetic demonstration")

    start = (70.0, 395.0)
    control = (160.0, 65.0)
    goal = (545.0, 90.0)

    try:
        for i in range(frame_count):
            t = i / (frame_count - 1)
            hand = _bezier(t, start, control, goal)
            frame = np.full((HEIGHT, WIDTH, 3), 245, dtype=np.uint8)
            cv2.circle(frame, tuple(map(int, goal)), 25, (0, 0, 255), -1)
            cv2.circle(frame, tuple(map(int, hand)), 12, (0, 255, 0), -1)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    output_dir = Path("outputs/visualizations")
    video_path = output_dir / "human_skill_demo.avi"
    write_human_demo(video_path)

    intent, perception = learn_motion_intent_from_video(
        str(video_path),
        hand_hsv=DEFAULT_EFFECTOR_HSV,
        target_hsv=DEFAULT_TARGET_HSV,
    )

    robot_start = (0.14, 0.86)
    robot_goal = (0.86, 0.16)

    # Case A: the transferred human strategy is valid and should be preferred.
    safe_plan = plan_from_robot_state(
        robot_start,
        robot_goal,
        demonstration_intent=intent,
        num_candidates=9,
        robot_config={"workspace": (0, 0, 1, 1), "max_lateral_offset": 0.45},
    )

    # Case B: place an obstacle on the transferred human trajectory. The same
    # learned skill is still proposed, but imagination should reject it.
    human_candidate = trajectory_from_demonstration(intent, robot_start, robot_goal)
    bx, by = human_candidate.waypoints[len(human_candidate.waypoints) // 2]
    obstacle = (
        max(0.0, bx - 0.06),
        max(0.0, by - 0.06),
        min(1.0, bx + 0.06),
        min(1.0, by + 0.06),
    )

    blocked_plan = plan_from_robot_state(
        robot_start,
        robot_goal,
        obstacles=[obstacle],
        demonstration_intent=intent,
        num_candidates=9,
        robot_config={"workspace": (0, 0, 1, 1), "max_lateral_offset": 0.45},
    )

    simulator = TabletopSimulator(
        effector_xy=robot_start,
        target_xy=robot_goal,
        obstacles=[obstacle],
    )
    execution = run_closed_loop(
        simulator,
        demonstration_intent=intent,
        max_attempts=2,
        num_candidates=9,
    )

    summary = {
        "perception": perception,
        "learned_path_points": len(intent.path_xy),
        "safe_scene_selected": safe_plan["selected_trajectory"]["trajectory_id"],
        "blocked_scene_selected": blocked_plan["selected_trajectory"]["trajectory_id"],
        "blocked_human_collision_risk": next(
            row["collision_risk"]
            for row in blocked_plan["ranking"]
            if row["trajectory_id"] == "human-demo"
        ),
        "closed_loop_execution_success": execution["success"],
        "closed_loop_selected_source": execution["history"][0]["selected_source"],
    }
    print(json.dumps(summary, indent=2))

    if summary["safe_scene_selected"] != "human-demo":
        raise SystemExit("Safe scene did not preserve the demonstrated human strategy")
    if summary["blocked_scene_selected"] == "human-demo":
        raise SystemExit("Planner blindly imitated a dangerous human trajectory")
    if not execution["success"]:
        raise SystemExit("Robot failed to adapt and execute a safe alternative")


if __name__ == "__main__":
    main()

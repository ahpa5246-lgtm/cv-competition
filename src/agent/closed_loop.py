"""Closed-loop visual planning agent.

Every attempt begins from a newly observed frame. Execution outcome is never
trusted as ground truth for success: the agent re-observes the scene through
OpenCV and verifies the visual goal before deciding whether to stop or replan.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.execution.protocol import VisualEnvironment
from src.pipeline import plan_from_robot_state
from src.verification.visual_verifier import verify_target_reached
from src.vision.scene_observer import observe_scene

DEFAULT_EFFECTOR_HSV = ((45, 120, 80), (85, 255, 255))
DEFAULT_TARGET_HSV = ((0, 140, 100), (10, 255, 255))
DEFAULT_OBSTACLE_HSV = ((100, 120, 80), (140, 255, 255))


def run_closed_loop(
    environment: VisualEnvironment,
    *,
    max_attempts: int = 3,
    num_candidates: int = 9,
    max_lateral_offset: float = 0.45,
    verification_tolerance: float = 0.065,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    target_xy = None

    for attempt in range(1, max_attempts + 1):
        pre_frame = environment.render()
        scene = observe_scene(
            pre_frame,
            effector_hsv=DEFAULT_EFFECTOR_HSV,
            target_hsv=DEFAULT_TARGET_HSV,
            obstacle_hsv=DEFAULT_OBSTACLE_HSV,
            frame_index=attempt - 1,
        )
        if scene.hand_xy is None or scene.target_xy is None:
            return {
                "success": False,
                "attempts": attempt - 1,
                "reason": "perception_failure",
                "history": history,
            }

        target_xy = scene.target_xy
        plan = plan_from_robot_state(
            scene.hand_xy,
            scene.target_xy,
            obstacles=scene.obstacle_boxes,
            robot_config={
                "workspace": (0.0, 0.0, 1.0, 1.0),
                "max_lateral_offset": max_lateral_offset,
            },
            num_candidates=num_candidates,
        )

        from src.core.types import RobotTrajectory

        selected_data = plan["selected_trajectory"]
        selected = RobotTrajectory(
            trajectory_id=selected_data["trajectory_id"],
            waypoints=[tuple(point) for point in selected_data["waypoints"]],
            source=selected_data["source"],
            metadata=dict(selected_data["metadata"]),
        )
        execution = environment.execute(selected)

        post_frame = environment.render()
        post_scene = observe_scene(
            post_frame,
            effector_hsv=DEFAULT_EFFECTOR_HSV,
            target_hsv=DEFAULT_TARGET_HSV,
            obstacle_hsv=DEFAULT_OBSTACLE_HSV,
            frame_index=attempt,
        )
        if post_scene.hand_xy is None or post_scene.target_xy is None:
            verified = {"success": False, "distance": float("inf"), "tolerance": verification_tolerance}
        else:
            verified = verify_target_reached(
                post_scene.hand_xy,
                post_scene.target_xy,
                tolerance=verification_tolerance,
            )

        history.append({
            "attempt": attempt,
            "observed_obstacles": scene.obstacle_boxes,
            "selected_trajectory": selected.trajectory_id,
            "predicted": plan["selected_prediction"],
            "execution": asdict(execution),
            "visual_verification": verified,
            "post_observed_obstacles": post_scene.obstacle_boxes,
        })

        if bool(verified["success"]):
            return {
                "success": True,
                "attempts": attempt,
                "reason": "visually_verified",
                "target_xy": target_xy,
                "history": history,
            }

    return {
        "success": False,
        "attempts": max_attempts,
        "reason": "max_attempts_exceeded",
        "target_xy": target_xy,
        "history": history,
    }

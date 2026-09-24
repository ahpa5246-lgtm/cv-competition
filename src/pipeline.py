"""End-to-end Watch → Understand → Retarget → Imagine → Select → Verify pipeline."""
from __future__ import annotations

from dataclasses import asdict

from src.core.types import BBox, Point2D, RobotTrajectory
from src.motion.representation import encode_motion
from src.prediction.future_model import HeuristicWorldModel, WorldModel
from src.prediction.rollout import rollout_candidates
from src.robot.mapping import map_to_robot
from src.robot.trajectory import generate_trajectory
from src.selection.candidate_scoring import score_candidates
from src.verification.visual_verifier import verify_target_reached
from src.vision.human_motion import extract_motion
from src.vision.object_tracking import track_colored_object
from src.vision.video_loader import load_video


def plan_from_robot_state(
    start_xy: Point2D,
    target_xy: Point2D,
    *,
    obstacles: list[BBox] | None = None,
    robot_config: dict | None = None,
    world_model: WorldModel | None = None,
    num_candidates: int = 5,
) -> dict:
    """Plan directly from the robot's currently observed state.

    This is the key entry point for recovery: after execution, OpenCV observes
    the actual robot pose and any changed obstacles, then replans from reality
    rather than from the original human demonstration.
    """
    obstacles = obstacles or []
    robot_config = robot_config or {"workspace": (0.0, 0.0, 1.0, 1.0)}
    world_model = world_model or HeuristicWorldModel()

    candidates = generate_trajectory(
        start_xy,
        target_xy,
        num_candidates=num_candidates,
        max_lateral_offset=float(robot_config.get("max_lateral_offset", 0.28)),
    )
    outcomes = rollout_candidates(world_model, candidates, target_xy, obstacles)
    ranked = score_candidates(outcomes)
    best_score = ranked[0]
    best: RobotTrajectory = next(
        candidate for candidate in candidates
        if candidate.trajectory_id == best_score.trajectory_id
    )

    return {
        "selected_trajectory": asdict(best),
        "selected_score": best_score.score,
        "selected_prediction": asdict(best_score.outcome),
        "ranking": [
            {
                "trajectory_id": item.trajectory_id,
                "score": item.score,
                "collision_risk": item.outcome.collision_risk,
                "success_probability": item.outcome.success_probability,
            }
            for item in ranked
        ],
    }


def plan_from_observation(
    observed_path: list[Point2D],
    target_xy: Point2D,
    frame_size: tuple[int, int],
    *,
    obstacles: list[BBox] | None = None,
    robot_config: dict | None = None,
    world_model: WorldModel | None = None,
    num_candidates: int = 5,
) -> dict:
    obstacles = obstacles or []
    robot_config = robot_config or {"workspace": (0.0, 0.0, 1.0, 1.0)}
    world_model = world_model or HeuristicWorldModel()

    intent = encode_motion(observed_path, target_xy, frame_size)
    start, goal, demonstrated_path = map_to_robot(intent, robot_config)

    plan = plan_from_robot_state(
        start,
        goal,
        obstacles=obstacles,
        robot_config=robot_config,
        world_model=world_model,
        num_candidates=num_candidates,
    )
    predicted_verification = verify_target_reached(
        tuple(plan["selected_prediction"]["predicted_final_xy"]),
        goal,
    )

    return {
        "intent": asdict(intent),
        "demonstrated_robot_path": demonstrated_path,
        **plan,
        "predicted_verification": predicted_verification,
    }


def run_pipeline(
    video_path: str,
    *,
    hand_hsv: tuple[tuple[int, int, int], tuple[int, int, int]],
    target_hsv: tuple[tuple[int, int, int], tuple[int, int, int]],
    obstacles: list[BBox] | None = None,
    robot_config: dict | None = None,
    max_frames: int | None = None,
) -> dict:
    """Run perception and planning from an ordinary human demonstration video.

    This function plans but does not claim physical success. Actual success must
    be established by post-execution visual verification in the closed-loop
    agent.
    """
    video = load_video(video_path, max_frames=max_frames)
    hand_track = track_colored_object(video.frames, *hand_hsv)
    target_track = track_colored_object(video.frames, *target_hsv)

    observed_path = [point for point in hand_track if point is not None]
    targets = [point for point in target_track if point is not None]
    if len(observed_path) < 2:
        raise RuntimeError("Could not recover enough hand/tool positions from the video")
    if not targets:
        raise RuntimeError("Could not find the target object in the video")

    flow = extract_motion(video.frames)
    result = plan_from_observation(
        observed_path,
        targets[-1],
        (video.width, video.height),
        obstacles=obstacles,
        robot_config=robot_config,
    )
    result["perception"] = {
        "frames": len(video.frames),
        "fps": video.fps,
        "tracked_points": len(observed_path),
        "optical_flow_samples": len(flow),
    }
    return result

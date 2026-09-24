"""Typed, serializable data structures shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Point2D = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class SceneState:
    frame_index: int
    timestamp_s: float
    hand_xy: Point2D | None = None
    target_xy: Point2D | None = None
    obstacle_boxes: list[BBox] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(slots=True)
class MotionIntent:
    task: str
    start_xy: Point2D
    goal_xy: Point2D
    path_xy: list[Point2D]
    coordinate_space: str = "normalized"


@dataclass(slots=True)
class RobotTrajectory:
    trajectory_id: str
    waypoints: list[Point2D]
    source: str = "retargeted-human-demonstration"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutOutcome:
    trajectory_id: str
    predicted_final_xy: Point2D
    target_distance: float
    collision_risk: float
    success_probability: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateScore:
    trajectory_id: str
    score: float
    outcome: RolloutOutcome


@dataclass(slots=True)
class ExecutionResult:
    trajectory_id: str
    final_xy: Point2D
    collision: bool
    reached_target: bool
    steps_executed: int
    metadata: dict[str, Any] = field(default_factory=dict)

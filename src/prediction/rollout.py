"""Evaluate candidate motions with a pluggable future/world model."""
from __future__ import annotations

from src.core.types import BBox, Point2D, RobotTrajectory, RolloutOutcome
from src.prediction.future_model import WorldModel


def rollout_candidates(
    model: WorldModel,
    candidates: list[RobotTrajectory],
    target_xy: Point2D,
    obstacles: list[BBox],
) -> list[RolloutOutcome]:
    return [model.predict(candidate, target_xy, obstacles) for candidate in candidates]

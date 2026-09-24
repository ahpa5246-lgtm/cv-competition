"""Simple candidate refinement hook.

A learned optimizer can replace this later. The MVP keeps refinement explicit
instead of pretending to contain a trained policy.
"""
from __future__ import annotations

from src.core.types import RobotTrajectory, RolloutOutcome


def refine_motion(
    candidate: RobotTrajectory,
    prediction: RolloutOutcome,
) -> RobotTrajectory:
    candidate.metadata = {
        **candidate.metadata,
        "last_predicted_success": prediction.success_probability,
        "last_collision_risk": prediction.collision_risk,
    }
    return candidate

"""World-model interface plus a deterministic planning baseline.

The heuristic model makes the full pipeline executable today. IRASim can be
connected later behind the same predict() contract and evaluated against it.
"""
from __future__ import annotations

from math import hypot
from typing import Protocol

from src.core.types import BBox, Point2D, RobotTrajectory, RolloutOutcome


class WorldModel(Protocol):
    def predict(
        self,
        trajectory: RobotTrajectory,
        target_xy: Point2D,
        obstacles: list[BBox],
    ) -> RolloutOutcome: ...


def _inside(point: Point2D, box: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


class HeuristicWorldModel:
    """Geometric baseline for testing the planner before IRASim integration."""

    def __init__(self, success_radius: float = 0.08) -> None:
        self.success_radius = success_radius

    def predict(
        self,
        trajectory: RobotTrajectory,
        target_xy: Point2D,
        obstacles: list[BBox],
    ) -> RolloutOutcome:
        final = trajectory.waypoints[-1]
        distance = hypot(final[0] - target_xy[0], final[1] - target_xy[1])
        collisions = sum(
            1
            for point in trajectory.waypoints
            if any(_inside(point, obstacle) for obstacle in obstacles)
        )
        collision_risk = collisions / max(len(trajectory.waypoints), 1)

        goal_quality = max(0.0, 1.0 - distance / max(self.success_radius * 3.0, 1e-8))
        success_probability = goal_quality * (1.0 - collision_risk)

        return RolloutOutcome(
            trajectory_id=trajectory.trajectory_id,
            predicted_final_xy=final,
            target_distance=distance,
            collision_risk=collision_risk,
            success_probability=success_probability,
            metadata={
                "model": "heuristic-baseline",
                "lateral_offset": float(trajectory.metadata.get("lateral_offset", 0.0)),
            },
        )


class IRASimAdapter:
    """Integration boundary for the research world model.

    Deliberately not faked: connect the actual IRASim inference stack here and
    return the same RolloutOutcome schema after visual/physical evaluation.
    """

    def predict(
        self,
        trajectory: RobotTrajectory,
        target_xy: Point2D,
        obstacles: list[BBox],
    ) -> RolloutOutcome:
        raise NotImplementedError(
            "IRASimAdapter is an explicit integration point; install/configure "
            "the real IRASim model before enabling it."
        )


def predict_future(
    model: WorldModel,
    trajectory: RobotTrajectory,
    target_xy: Point2D,
    obstacles: list[BBox],
) -> RolloutOutcome:
    return model.predict(trajectory, target_xy, obstacles)

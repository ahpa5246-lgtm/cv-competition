"""Minimal deterministic tabletop simulator for closed-loop agent tests.

Coordinates live in a normalized [0, 1] x [0, 1] workspace. The simulator is
not a physics engine; it exists to test perception → planning → execution →
visual re-observation → recovery logic before connecting a robotics simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

import cv2
import numpy as np

from src.core.types import BBox, ExecutionResult, Point2D, RobotTrajectory


def _inside(point: Point2D, box: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class TabletopSimulator:
    effector_xy: Point2D = (0.10, 0.82)
    target_xy: Point2D = (0.88, 0.18)
    obstacles: list[BBox] = field(default_factory=list)
    width: int = 640
    height: int = 480
    target_tolerance: float = 0.055

    _attempt: int = 0
    _scheduled_obstacles: dict[int, list[BBox]] = field(default_factory=dict)

    def schedule_obstacles(self, attempt: int, obstacles: list[BBox]) -> None:
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        self._scheduled_obstacles[attempt] = list(obstacles)

    def execute(self, trajectory: RobotTrajectory) -> ExecutionResult:
        self._attempt += 1
        if self._attempt in self._scheduled_obstacles:
            self.obstacles = list(self._scheduled_obstacles[self._attempt])

        last_safe = self.effector_xy
        collided = False
        steps = 0

        for point in trajectory.waypoints:
            steps += 1
            if not (0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0):
                collided = True
                break
            if any(_inside(point, obstacle) for obstacle in self.obstacles):
                collided = True
                break
            last_safe = point

        self.effector_xy = last_safe
        distance = hypot(
            self.effector_xy[0] - self.target_xy[0],
            self.effector_xy[1] - self.target_xy[1],
        )
        return ExecutionResult(
            trajectory_id=trajectory.trajectory_id,
            final_xy=self.effector_xy,
            collision=collided,
            reached_target=(not collided and distance <= self.target_tolerance),
            steps_executed=steps,
            metadata={"attempt": self._attempt, "target_distance": distance},
        )

    def render(self) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), 245, dtype=np.uint8)

        def px(point: Point2D) -> tuple[int, int]:
            return int(point[0] * self.width), int(point[1] * self.height)

        for box in self.obstacles:
            x1, y1, x2, y2 = box
            cv2.rectangle(
                frame,
                (int(x1 * self.width), int(y1 * self.height)),
                (int(x2 * self.width), int(y2 * self.height)),
                (255, 0, 0),
                -1,
            )

        cv2.circle(frame, px(self.target_xy), 23, (0, 0, 255), -1)
        cv2.circle(frame, px(self.effector_xy), 12, (0, 255, 0), -1)
        return frame

"""AWS rollout-service boundary.

The local planner remains usable without AWS. For the competition deployment,
heavy world-model inference can be hosted on AWS and exposed through an HTTPS
endpoint. This client is intentionally transport-only: it does not fake a
deployed service.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen

from src.core.types import BBox, Point2D, RobotTrajectory, RolloutOutcome


class AWSRolloutClient:
    def __init__(self, endpoint_url: str, timeout_s: float = 30.0) -> None:
        if not endpoint_url.startswith("https://"):
            raise ValueError("endpoint_url must use HTTPS")
        self.endpoint_url = endpoint_url
        self.timeout_s = timeout_s

    def predict(
        self,
        trajectory: RobotTrajectory,
        target_xy: Point2D,
        obstacles: list[BBox],
    ) -> RolloutOutcome:
        payload = json.dumps({
            "trajectory_id": trajectory.trajectory_id,
            "waypoints": trajectory.waypoints,
            "target_xy": target_xy,
            "obstacles": obstacles,
        }).encode("utf-8")

        request = Request(
            self.endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))

        return RolloutOutcome(
            trajectory_id=body["trajectory_id"],
            predicted_final_xy=tuple(body["predicted_final_xy"]),
            target_distance=float(body["target_distance"]),
            collision_risk=float(body["collision_risk"]),
            success_probability=float(body["success_probability"]),
            metadata={"model": body.get("model", "aws-rollout-service")},
        )

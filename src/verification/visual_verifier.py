"""Verify whether the observed post-action scene satisfies the visual goal."""
from __future__ import annotations

from math import hypot
from src.core.types import Point2D


def verify_target_reached(
    observed_xy: Point2D,
    target_xy: Point2D,
    tolerance: float = 0.08,
) -> dict[str, float | bool]:
    distance = hypot(observed_xy[0] - target_xy[0], observed_xy[1] - target_xy[1])
    return {"success": distance <= tolerance, "distance": distance, "tolerance": tolerance}

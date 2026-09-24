"""Transfer the *shape* of a human-demonstrated path to a new robot scene.

The representation is embodiment-agnostic: points are expressed relative to
the demonstrated start→goal axis, then reconstructed around the robot's
currently observed start→goal axis.
"""
from __future__ import annotations

from math import hypot

from src.core.types import MotionIntent, Point2D, RobotTrajectory


def _resample_path(points: list[Point2D], count: int) -> list[Point2D]:
    if count < 2:
        raise ValueError("count must be >= 2")
    if len(points) < 2:
        raise ValueError("At least two points are required")

    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + hypot(b[0] - a[0], b[1] - a[1]))

    total = cumulative[-1]
    if total <= 1e-9:
        return [points[0]] * count

    result: list[Point2D] = []
    segment = 0
    for i in range(count):
        target_distance = total * i / (count - 1)
        while segment < len(cumulative) - 2 and cumulative[segment + 1] < target_distance:
            segment += 1

        a, b = points[segment], points[segment + 1]
        d0, d1 = cumulative[segment], cumulative[segment + 1]
        local_t = 0.0 if d1 <= d0 else (target_distance - d0) / (d1 - d0)
        result.append((
            a[0] + (b[0] - a[0]) * local_t,
            a[1] + (b[1] - a[1]) * local_t,
        ))
    return result


def retarget_path_shape(
    demonstrated_path: list[Point2D],
    demonstrated_start: Point2D,
    demonstrated_goal: Point2D,
    robot_start: Point2D,
    robot_goal: Point2D,
    *,
    num_waypoints: int = 25,
) -> list[Point2D]:
    """Similarity-transfer a demonstrated path to a new start/goal pair."""
    demo_dx = demonstrated_goal[0] - demonstrated_start[0]
    demo_dy = demonstrated_goal[1] - demonstrated_start[1]
    demo_length = hypot(demo_dx, demo_dy)
    if demo_length <= 1e-8:
        raise ValueError("Demonstrated start and goal must differ")

    robot_dx = robot_goal[0] - robot_start[0]
    robot_dy = robot_goal[1] - robot_start[1]
    robot_length = hypot(robot_dx, robot_dy)
    if robot_length <= 1e-8:
        raise ValueError("Robot start and goal must differ")

    demo_e = (demo_dx / demo_length, demo_dy / demo_length)
    demo_n = (-demo_e[1], demo_e[0])
    robot_e = (robot_dx / robot_length, robot_dy / robot_length)
    robot_n = (-robot_e[1], robot_e[0])

    transferred: list[Point2D] = []
    for point in demonstrated_path:
        rx = point[0] - demonstrated_start[0]
        ry = point[1] - demonstrated_start[1]
        longitudinal = (rx * demo_e[0] + ry * demo_e[1]) / demo_length
        lateral = (rx * demo_n[0] + ry * demo_n[1]) / demo_length

        transferred.append((
            robot_start[0]
            + longitudinal * robot_length * robot_e[0]
            + lateral * robot_length * robot_n[0],
            robot_start[1]
            + longitudinal * robot_length * robot_e[1]
            + lateral * robot_length * robot_n[1],
        ))

    if not transferred:
        transferred = [robot_start, robot_goal]
    else:
        transferred[0] = robot_start
        # The human may stop a few pixels before the detected target. For robot
        # execution we explicitly preserve the demonstrated *shape* but finish
        # at the requested robot goal.
        transferred.append(robot_goal)

    sampled = _resample_path(transferred, num_waypoints)
    sampled[0] = robot_start
    sampled[-1] = robot_goal
    return sampled


def trajectory_from_demonstration(
    intent: MotionIntent,
    robot_start: Point2D,
    robot_goal: Point2D,
    *,
    num_waypoints: int = 25,
) -> RobotTrajectory:
    waypoints = retarget_path_shape(
        intent.path_xy,
        intent.start_xy,
        intent.goal_xy,
        robot_start,
        robot_goal,
        num_waypoints=num_waypoints,
    )
    return RobotTrajectory(
        trajectory_id="human-demo",
        waypoints=waypoints,
        source="human-demonstration-transfer",
        metadata={
            "demonstration_fidelity": 1.0,
            "task": intent.task,
            "lateral_offset": 0.0,
        },
    )

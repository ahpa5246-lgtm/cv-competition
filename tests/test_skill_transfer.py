from math import hypot

from src.core.types import MotionIntent
from src.robot.skill_transfer import retarget_path_shape, trajectory_from_demonstration


def test_path_shape_retargets_endpoints_and_curvature():
    demo_start = (0.10, 0.80)
    demo_goal = (0.90, 0.20)
    demo_path = [
        demo_start,
        (0.30, 0.45),
        (0.60, 0.20),
        demo_goal,
    ]

    robot_start = (0.85, 0.85)
    robot_goal = (0.15, 0.15)
    transferred = retarget_path_shape(
        demo_path,
        demo_start,
        demo_goal,
        robot_start,
        robot_goal,
        num_waypoints=25,
    )

    assert transferred[0] == robot_start
    assert transferred[-1] == robot_goal
    assert len(transferred) == 25

    # A non-straight human demonstration must remain non-straight after
    # retargeting rather than collapsing into direct interpolation.
    sx, sy = robot_start
    gx, gy = robot_goal
    max_deviation = 0.0
    line_len = hypot(gx - sx, gy - sy)
    for x, y in transferred[1:-1]:
        cross = abs((gx - sx) * (sy - y) - (sx - x) * (gy - sy))
        max_deviation = max(max_deviation, cross / line_len)
    assert max_deviation > 0.05


def test_demonstration_trajectory_has_explicit_provenance():
    intent = MotionIntent(
        task="reach_target",
        start_xy=(0.1, 0.8),
        goal_xy=(0.9, 0.2),
        path_xy=[(0.1, 0.8), (0.5, 0.3), (0.9, 0.2)],
    )
    trajectory = trajectory_from_demonstration(intent, (0.2, 0.9), (0.8, 0.1))
    assert trajectory.source == "human-demonstration-transfer"
    assert trajectory.metadata["demonstration_fidelity"] == 1.0

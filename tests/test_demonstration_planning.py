from src.core.types import MotionIntent
from src.pipeline import plan_from_robot_state
from src.robot.skill_transfer import trajectory_from_demonstration


def _intent():
    return MotionIntent(
        task="reach_target",
        start_xy=(0.10, 0.82),
        goal_xy=(0.88, 0.18),
        path_xy=[
            (0.10, 0.82),
            (0.18, 0.48),
            (0.35, 0.20),
            (0.62, 0.08),
            (0.88, 0.18),
        ],
    )


def test_safe_human_demonstration_is_preferred():
    intent = _intent()
    start = (0.14, 0.86)
    goal = (0.86, 0.16)

    plan = plan_from_robot_state(
        start,
        goal,
        demonstration_intent=intent,
        num_candidates=9,
        robot_config={"workspace": (0, 0, 1, 1), "max_lateral_offset": 0.45},
    )

    assert plan["selected_trajectory"]["trajectory_id"] == "human-demo"
    assert plan["selected_trajectory"]["source"] == "human-demonstration-transfer"


def test_dangerous_human_demonstration_is_rejected():
    intent = _intent()
    start = (0.14, 0.86)
    goal = (0.86, 0.16)
    demo = trajectory_from_demonstration(intent, start, goal)
    x, y = demo.waypoints[len(demo.waypoints) // 2]
    obstacle = [(x - 0.055, y - 0.055, x + 0.055, y + 0.055)]

    plan = plan_from_robot_state(
        start,
        goal,
        obstacles=obstacle,
        demonstration_intent=intent,
        num_candidates=9,
        robot_config={"workspace": (0, 0, 1, 1), "max_lateral_offset": 0.45},
    )

    assert plan["selected_trajectory"]["trajectory_id"] != "human-demo"
    assert plan["selected_prediction"]["collision_risk"] == 0.0

    human_row = next(row for row in plan["ranking"] if row["trajectory_id"] == "human-demo")
    assert human_row["collision_risk"] > 0.0

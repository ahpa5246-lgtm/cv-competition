from src.pipeline import plan_from_observation


def test_world_model_planner_selects_collision_free_candidate():
    observed = [(10.0, 80.0), (30.0, 65.0), (50.0, 50.0)]
    target = (90.0, 20.0)
    obstacle = [(0.42, 0.38, 0.62, 0.66)]

    result = plan_from_observation(
        observed,
        target,
        (100, 100),
        obstacles=obstacle,
        robot_config={
            "workspace": (0.0, 0.0, 1.0, 1.0),
            "max_lateral_offset": 0.45,
        },
        num_candidates=7,
    )

    assert result["verification"]["success"] is True
    assert result["selected_prediction"]["collision_risk"] == 0.0

    # 7 generated alternatives + 1 transferred human demonstration.
    assert len(result["ranking"]) == 8
    sources = {row["source"] for row in result["ranking"]}
    assert "human-demonstration-transfer" in sources
    assert "generated-alternative" in sources

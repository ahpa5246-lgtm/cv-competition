from simulation.tabletop import TabletopSimulator
from src.agent.closed_loop import run_closed_loop


def test_agent_recovers_after_surprise_obstacle():
    simulator = TabletopSimulator(
        effector_xy=(0.10, 0.82),
        target_xy=(0.88, 0.18),
    )
    simulator.schedule_obstacles(
        attempt=1,
        obstacles=[(0.43, 0.43, 0.60, 0.64)],
    )

    result = run_closed_loop(simulator, max_attempts=3)

    assert result["success"] is True
    assert result["attempts"] == 2
    assert result["history"][0]["execution"]["collision"] is True
    assert len(result["history"][1]["observed_obstacles"]) == 1
    assert result["history"][1]["visual_verification"]["success"] is True

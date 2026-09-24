"""Score imagined outcomes and select the action to execute."""
from __future__ import annotations

from src.core.types import CandidateScore, RolloutOutcome


def score_candidates(
    predictions: list[RolloutOutcome],
    *,
    success_weight: float = 2.0,
    collision_weight: float = 3.0,
    distance_weight: float = 1.0,
) -> list[CandidateScore]:
    scored = [
        CandidateScore(
            trajectory_id=outcome.trajectory_id,
            score=(
                success_weight * outcome.success_probability
                - collision_weight * outcome.collision_risk
                - distance_weight * outcome.target_distance
            ),
            outcome=outcome,
        )
        for outcome in predictions
    ]
    return sorted(scored, key=lambda item: item.score, reverse=True)


def select_best(predictions: list[RolloutOutcome]) -> CandidateScore:
    if not predictions:
        raise ValueError("No predictions to score")
    return score_candidates(predictions)[0]

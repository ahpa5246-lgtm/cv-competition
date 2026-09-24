"""Score imagined outcomes and select the action to execute."""
from __future__ import annotations

from src.core.types import CandidateScore, RolloutOutcome


def score_candidates(
    predictions: list[RolloutOutcome],
    *,
    success_weight: float = 2.0,
    collision_weight: float = 3.0,
    distance_weight: float = 1.0,
    deviation_weight: float = 0.15,
    demonstration_weight: float = 0.08,
) -> list[CandidateScore]:
    scored = []
    for outcome in predictions:
        lateral_offset = abs(float(outcome.metadata.get("lateral_offset", 0.0)))
        demonstration_fidelity = float(
            outcome.metadata.get("demonstration_fidelity", 0.0)
        )
        score = (
            success_weight * outcome.success_probability
            - collision_weight * outcome.collision_risk
            - distance_weight * outcome.target_distance
            - deviation_weight * lateral_offset
            + demonstration_weight * demonstration_fidelity
        )
        scored.append(
            CandidateScore(
                trajectory_id=outcome.trajectory_id,
                score=score,
                outcome=outcome,
            )
        )
    return sorted(scored, key=lambda item: item.score, reverse=True)


def select_best(predictions: list[RolloutOutcome]) -> CandidateScore:
    if not predictions:
        raise ValueError("No predictions to score")
    return score_candidates(predictions)[0]

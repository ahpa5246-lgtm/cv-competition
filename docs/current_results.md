# Current reproducible results

These results are **synthetic engineering evidence only**. They validate the
software architecture and agent loop. They must not be presented as real-robot
performance.

## Environment

- Python 3.11
- OpenCV 5.0.0
- GitHub Actions Ubuntu runner
- Deterministic geometric world-model baseline
- Tabletop 2D simulator

## CI evidence

A successful CI run validated:

- 5 unit/integration tests passed.
- OpenCV version assertion passed.
- Synthetic perception/planning demo passed.
- Direct-vs-planner benchmark passed.
- Closed-loop surprise-obstacle recovery demo passed.

### Planning benchmark

50 synthetic episodes were generated so that the direct path intersects an
obstacle.

| Metric | Result |
|---|---:|
| Direct success rate | 0.00 |
| Direct collision rate | 1.00 |
| Planned success rate | 1.00 |
| Planned collision rate | 0.00 |

This benchmark is intentionally adversarial and therefore **not a general
success-rate claim**. Its purpose is to verify that candidate imagination and
selection can choose a collision-free alternative when one exists.

### Closed-loop recovery trace

The recovery demo begins with no visible obstacle.

**Attempt 1**

- Observed obstacles: none.
- Selected trajectory: central/direct candidate (`lateral_offset = 0.0`).
- Predicted success probability: approximately 1.0.
- A surprise obstacle appears only after planning.
- Execution collides/stops after 12 trajectory steps.
- Visual verification fails.
- The post-action OpenCV observation detects the newly appeared obstacle.

**Attempt 2**

- Planning starts from the effector's newly observed actual position.
- The new obstacle is included in scene state.
- The planner selects a different trajectory (`lateral_offset = -0.225`).
- Predicted collision risk: 0.
- Execution reaches the target without collision.
- Post-action OpenCV verification reports success with observed target distance
  0 in the rendered frame.

This is the first repository-level proof of:

**Predict → Act → Observe contradiction → Update world state → Replan → Verify**

## What this does not prove

It does not yet establish:

- real robot performance;
- manipulation/contact physics;
- grasp stability;
- robustness to unconstrained real-world appearance;
- StaMo integration;
- IRASim integration;
- deployed AWS inference.

Those remain separate milestones and must be benchmarked rather than claimed.

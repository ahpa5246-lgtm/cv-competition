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

### Recovery benchmark

A second CI benchmark ran 20 deterministic surprise-obstacle episodes
(`seed = 42`) and compared one-shot open-loop execution with visual
re-observation/replanning.

| Metric | Result |
|---|---:|
| Open-loop success rate | 0.00 |
| Open-loop collision rate | 1.00 |
| Closed-loop success rate | 1.00 |
| Closed-loop mean attempts | 2.00 |

The scenarios are deliberately constructed so the obstacle appears after the
initial plan and intersects the initially preferred direct route. These values
therefore demonstrate recovery-loop correctness under that controlled
condition; they are **not** estimates of general real-world reliability.

### Human skill-transfer trace

The end-to-end skill-transfer CI demo generated a 60-frame human motion video
at 20 FPS. OpenCV recovered all 60 tracked hand positions and 59 optical-flow
transitions.

The learned curved path was then transferred into a different robot start/goal
configuration.

| Condition | Planner decision |
|---|---|
| Transferred human path is safe | selected `human-demo` |
| New obstacle blocks transferred human path | selected `candidate-04` |
| Predicted collision risk for blocked `human-demo` | 0.12 |
| Closed-loop execution of safe alternative | success |
| Executed trajectory provenance | `generated-alternative` |

This is controlled synthetic evidence for the intended rule:

**preserve useful human motion structure when safe; reject or adapt it when the
current scene makes imitation unsafe.**

### YAML experiment harness

The CI harness exercised the same workflow intended for real recordings:

- loaded a human demonstration video from file;
- tracked 60 motion points with OpenCV;
- loaded a separate robot-scene image;
- detected 1 obstacle;
- transferred the learned human motion;
- selected `human-demo` in this safe test scene;
- wrote a structured `report.json`;
- wrote an annotated `overlay.png`.

This validates the experiment/evidence plumbing needed to move from synthetic
unit tests to manually recorded videos without editing the planning code.

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

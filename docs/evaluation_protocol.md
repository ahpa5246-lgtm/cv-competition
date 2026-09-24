# Evaluation protocol

The competition demo must demonstrate a measurable benefit from **imagining
before acting**, not merely successful object detection.

## Core hypotheses

H1. Task-centric retargeting transfers a human demonstration better than
copying the raw observed path.

H2. World-model planning reduces collisions and increases manipulation success
relative to direct execution.

H3. Closed-loop visual verification recovers failures that an open-loop policy
would leave unresolved.

## Required experiment matrix

| System | Perception | Retarget | Future model | Verify |
|---|---|---|---|---|
| B0 direct | OpenCV | direct | none | no |
| B1 retarget | OpenCV | yes | none | no |
| B2 planner | OpenCV | yes | heuristic | yes |
| M1 research | OpenCV/learned | yes | IRASim | yes |

If StaMo is integrated, add an ablation replacing the baseline motion
representation while keeping the rest fixed.

## Test conditions

For each task, vary at minimum:
- target position;
- demonstration viewpoint;
- obstacle position;
- initial robot/end-effector position;
- object appearance where supported.

Recommended initial tasks:
1. reach target;
2. push object to target;
3. pick-and-place once grasp simulation is available.

## Metrics

- **Task success rate**: successful episodes / total episodes.
- **Collision rate**: episodes with any obstacle collision / total.
- **Final target distance**.
- **Planning latency** and perception latency.
- **Prediction calibration**: whether predicted success matches observed success.
- **Recovery success rate** after verification detects a failed outcome.

Report mean, dispersion, sample count, and failure examples. Do not report only
the best demo.

## Competition evidence

The final report/video should show:
1. the same scene with direct imitation failing or colliding;
2. candidate imagined outcomes;
3. the selected trajectory;
4. execution;
5. camera-based verification;
6. quantitative aggregate results across many scenes.

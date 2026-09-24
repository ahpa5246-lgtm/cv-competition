# Research-model integration boundaries

This repository avoids claiming integrations that are not yet running.

## StaMo direction

Target insertion point: `src/motion/representation.py`.

The current baseline converts tracked visual positions into a task-centric
normalized representation. A StaMo-derived adapter should expose the same
high-level intent/path contract so it can be compared against the baseline.

Success criterion: improved transfer success under scene/viewpoint variation,
not simply successful model inference.

## IRASim direction

Target insertion point: `src/prediction/future_model.py::IRASimAdapter`.

The adapter should:
1. condition the model on the current visual state and candidate robot action;
2. generate/obtain the future rollout;
3. evaluate goal achievement, collisions, and uncertainty from that rollout;
4. return a `RolloutOutcome`.

Success criterion: better candidate ranking than the deterministic heuristic
baseline on held-out scenes.

## Rule

Do not merge an adapter into the competition branch until:
- inference is reproducible;
- model/checkpoint provenance is documented;
- license constraints are recorded;
- an automated or scripted benchmark compares it with the baseline.

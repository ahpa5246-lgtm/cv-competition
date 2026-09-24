# Watch → Imagine → Act

A computer-vision + robotics system that learns task intent from human video,
retargets it to a robot, imagines multiple candidate futures before execution,
selects a safer successful trajectory, acts, **re-observes the world**, and
replans until the visual goal is verified.

## Problem

Teaching manipulation skills to robots often requires robot-specific
demonstrations, teleoperation, manually designed trajectories, or costly
physical trial-and-error. A human demonstration also cannot simply be copied:
the robot has different kinematics and may face a changed scene.

The project implements:

**Watch → Understand → Retarget → Imagine → Select → Act → Observe → Verify → Replan**

The first competition scope remains deliberately narrow and measurable:
**tabletop manipulation from human demonstrations**.

## Why this is a computer-vision project

OpenCV 5 is not used merely to read a video. The perception layer owns frame
processing, temporal motion, object/hand localization, obstacle discovery,
camera-to-table geometry, coordinate transforms, and post-action visual
verification.

The current MVP implements transparent OpenCV baselines:
HSV segmentation, contour geometry, dense Farneback optical flow and
homography estimation. OpenCV DNN detectors/segmenters can replace the
baseline perception behind stable interfaces.

## Architecture

```
Human demonstration video
          │
          ▼
┌───────────────────────────┐
│ OpenCV 5 perception       │
│ objects / motion / scene  │
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│ Task-centric intent       │
│ StaMo adapter planned     │
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│ Human → robot retargeting │
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│ Candidate trajectories    │
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│ Imagine future outcomes   │
│ heuristic → IRASim/AWS    │
└────────────┬──────────────┘
             ▼
       select + execute
             │
             ▼
┌───────────────────────────┐
│ OpenCV re-observation     │
│ actual pose + new hazards │
└────────────┬──────────────┘
       success? ── yes ──► stop
             │ no
             └────────────► replan from observed reality
```

## What is implemented now

- OpenCV 5 video ingestion.
- OpenCV HSV/contour localization for effector, target and obstacles.
- Dense Farneback optical flow for temporal motion evidence.
- OpenCV homography estimation for camera-to-workspace geometry.
- Normalized task-centric motion representation.
- Human-to-robot coordinate retargeting.
- **Transfer of the demonstrated human path shape into a new robot start/goal configuration.**
- A provenance-tagged `human-demo` candidate that competes with generated alternatives.
- The planner prefers the human strategy when safe and rejects it when imagined risk is higher.
- Multiple candidate trajectory generation.
- Pluggable world-model interface.
- Deterministic geometric future-model baseline.
- Explicit IRASim integration boundary.
- Candidate scoring with success, collision, distance and deviation costs.
- Tabletop execution simulator.
- **Visual post-execution re-observation.**
- **Automatic replanning from the robot's actually observed position.**
- **Recovery after an obstacle appears after planning.**
- Synthetic direct-vs-planner benchmark.
- Synthetic open-loop-vs-closed-loop recovery benchmark.
- AWS rollout-service client boundary.
- Unit tests and GitHub Actions CI.

## Human skill-transfer demo

Run:

```bash
python -m simulation.skill_transfer_demo
```

This demo generates a human demonstration video with a curved reaching motion.
OpenCV recovers the temporal path and target, converts them into a normalized
motion intent, and transfers that path geometry to a robot scene with a
different start/goal configuration.

Two cases are evaluated:

1. **Safe transfer:** the demonstrated strategy is valid, so `human-demo` is
   selected over generic alternatives.
2. **Unsafe transfer:** an obstacle is placed on the transferred human path.
   The same human strategy is still proposed, but the future-model score marks
   collision risk and the planner chooses a generated safe alternative.

The intended behavior is therefore **learn from the human, but do not blindly
imitate the human**.

See `docs/skill_transfer.md`.

## The critical recovery demo

Run:

```bash
python -m simulation.closed_loop_demo
```

The first camera observation contains no obstacle. The agent therefore plans a
direct trajectory. Immediately before execution, a new obstacle appears.

The first execution is interrupted. The agent does **not** trust its earlier
prediction. It captures a new image, OpenCV detects the new obstacle and the
actual effector location, then the planner generates new trajectories from that
observed state. A safe route is selected and execution continues until the
post-action image verifies the goal.

This directly demonstrates that visual evidence changes the next action.

See `docs/closed_loop_agent.md`.

## Real-video experiment harness

Once you have a recording, no code changes are required for a baseline test.

Copy `configs/real_video.example.yaml`, set:

- the human demonstration video;
- one robot-scene image/frame;
- HSV ranges for the human hand/tool, robot effector, target and obstacles;
- candidate-planning settings.

Then run:

```bash
python -m src.experiment.cli --config path/to/experiment.yaml
```

The experiment writes a machine-readable `report.json` and an
`overlay.png`. The report records OpenCV/Python versions, perception counts,
the learned human intent, the transferred human trajectory, the observed robot
scene, candidate ranking, selected trajectory and provenance.

This creates a repeatable evidence format that can later compare the OpenCV
baseline, StaMo-derived representations and IRASim world-model rollouts on the
same recordings.

See `docs/real_video_experiments.md`.

## Quick start

Requires Python 3.11+ and OpenCV 5.0.0.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .

python -m simulation.synthetic_demo
python -m simulation.closed_loop_demo
python -m simulation.skill_transfer_demo
python -m simulation.experiment_harness_demo
python -m simulation.benchmark --episodes 200
python -m simulation.recovery_benchmark --episodes 50
python -m pytest -q
```

All synthetic numbers are engineering validation of orchestration and planning,
not claims about physical-robot performance.

## Evaluation strategy

The final project should compare:

1. Direct retargeted imitation without future planning.
2. Candidate planning with the geometric baseline.
3. Candidate planning with IRASim.
4. Open-loop execution versus closed-loop visual verification/recovery.
5. Baseline motion representation versus a StaMo-derived learned representation.

Primary metrics include task success rate, collision rate, final target
distance, planning latency, recovery success rate, replans per episode and
prediction calibration.

See:
- `docs/evaluation_protocol.md`
- `docs/closed_loop_agent.md`
- `docs/skill_transfer.md`
- `docs/real_video_experiments.md`
- `docs/research_integration.md`
- `docs/aws_architecture.md`

## Repository layout

```
configs/
data/
docs/
simulation/
src/
  agent/                  observe → plan → act → verify → replan
  cloud/                  AWS rollout-service boundary
  core/                   shared typed schemas
  execution/              simulator/robot execution contract
  experiment/             YAML experiment runner + evidence overlays
  vision/                 OpenCV perception + geometry
  motion/                 task/motion representation
  robot/                  retargeting + trajectory generation
  prediction/             world-model interface and rollouts
  selection/              scoring and refinement
  verification/           visual goal verification
tests/
outputs/
```

## Competition thesis

The contribution is not merely object detection and not simply connecting two
research repositories. The system uses vision twice: first to learn and plan
from observed human behavior, and again after action to determine whether the
world changed and what the robot should do next.

**Perception is part of the control loop, not a visualization layer.**

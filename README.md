# Watch → Imagine → Act

A computer-vision + robotics system that learns task intent from human video,
retargets it to a robot, imagines multiple candidate futures before execution,
selects a safer successful trajectory, acts, and visually verifies the result.

## Problem

Teaching manipulation skills to robots often requires robot-specific
demonstrations, teleoperation, manually designed trajectories, or costly
physical trial-and-error. A human demonstration also cannot simply be copied:
the robot has different kinematics and may face a changed scene.

This project investigates a stronger loop:

**Watch → Understand → Retarget → Imagine → Select → Act → Verify**

The first competition scope is deliberately narrow and measurable:
**tabletop manipulation from human demonstrations**.

## Why this is a computer-vision project

OpenCV 5 is not used merely to read a video. The perception layer is designed
to own frame processing, temporal motion, object/hand localization, geometry,
coordinate transforms, and post-action visual verification. The current MVP
implements transparent OpenCV baselines (HSV segmentation + optical flow).
OpenCV DNN detectors/segmenters can replace those baselines behind the same
interfaces.

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
│ Task-centric motion       │
│ representation            │
│ (StaMo adapter planned)   │
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
│ World-model rollouts      │
│ heuristic baseline now    │
│ IRASim adapter planned    │
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│ Score / select / refine   │
└────────────┬──────────────┘
             ▼
       robot / simulator
             │
             ▼
┌───────────────────────────┐
│ OpenCV visual verification│
└───────────────────────────┘
```

## What is implemented now

- OpenCV 5 video ingestion.
- OpenCV HSV segmentation baseline for hand/tool and target localization.
- Dense Farneback optical flow for temporal motion evidence.
- Normalized task-centric motion representation.
- Cross-workspace human-to-robot coordinate retargeting.
- Generation of multiple robot trajectory candidates.
- Pluggable world-model interface.
- Deterministic geometric world-model baseline.
- Explicit IRASim adapter boundary (not a fake implementation).
- Candidate scoring by predicted success, collision risk, and target distance.
- Post-action goal verification.
- Reproducible synthetic end-to-end demonstration and unit tests.

## Research integrations

The repository intentionally separates **working baseline** from **research
model integration**.

- **StaMo direction:** upgrade task/motion representation from handcrafted
  tracking to a learned compact state/action representation.
- **IRASim direction:** replace the heuristic future model with real visual
  future rollouts, then evaluate whether those rollouts improve action
  selection.
- Neither integration is claimed as complete until real inference is wired and
  benchmarked.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m simulation.synthetic_demo
pytest -q
```

The synthetic demo writes a planning visualization under
`outputs/visualizations/` and prints the candidate ranking.

## Evaluation strategy

The system must beat meaningful baselines, not just produce a convincing demo.

1. Direct retargeted imitation without world-model planning.
2. Candidate planning with the geometric baseline.
3. Candidate planning with the IRASim adapter.
4. Optional learned state/action representation versus the OpenCV baseline.

Primary metrics: task success rate, collision rate, final target distance,
planning latency, recovery success, and calibration between predicted and
observed success.

See `docs/evaluation_protocol.md`.

## Repository layout

```
configs/                 experiment configuration
data/                    demonstrations and processed data
docs/                    architecture and evaluation protocol
simulation/              reproducible simulator/synthetic demos
src/
  core/                  shared typed schemas
  vision/                OpenCV perception
  motion/                task/motion representation
  robot/                 retargeting + trajectory generation
  prediction/            world-model interface and rollouts
  selection/             scoring and refinement
  verification/          visual closed-loop verification
tests/                    unit and pipeline tests
outputs/                  predictions and visualizations
```

## Competition thesis

The project is not "YOLO + robot" and not "StaMo + IRASim" as two names glued
together. The contribution is a complete decision loop in which **visual
evidence changes the robot's next action**: learn intent from a human
demonstration, adapt it to a different embodiment and scene, imagine candidate
consequences, choose, execute, observe the result, and recover when necessary.

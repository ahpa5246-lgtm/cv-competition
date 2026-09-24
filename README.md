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

OpenCV 5 is not used merely to read a video. The perception layer owns frame
processing, temporal motion, object/hand localization, camera-to-table
geometry, coordinate transforms, and post-action visual verification.

The current MVP implements transparent OpenCV baselines:
HSV segmentation, dense Farneback optical flow, contour geometry and
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
│ camera → table geometry   │
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
│ local baseline now        │
│ IRASim/AWS path planned   │
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
- OpenCV homography estimation for camera-to-workspace geometry.
- Normalized task-centric motion representation.
- Cross-workspace human-to-robot coordinate retargeting.
- Generation of multiple robot trajectory candidates.
- Pluggable world-model interface.
- Deterministic geometric world-model baseline.
- Explicit IRASim adapter boundary (not a fake implementation).
- Candidate scoring by predicted success, collision risk, and target distance.
- Post-action visual goal verification.
- Reproducible synthetic end-to-end demo.
- Synthetic direct-vs-planner ablation benchmark.
- AWS rollout-service client boundary.
- Unit tests and GitHub Actions CI.

## Research integrations

The repository intentionally separates **working baseline** from **research
model integration**.

- **StaMo direction:** upgrade task/motion representation from handcrafted
  tracking to a learned compact state/action representation.
- **IRASim direction:** replace the heuristic future model with real visual
  future rollouts, then evaluate whether those rollouts improve action
  selection.
- **AWS direction:** host compute-heavy counterfactual rollouts while keeping
  latency-sensitive OpenCV perception local.
- None of these integrations is claimed as complete until real inference or
  infrastructure is wired and benchmarked.

## Quick start

Requires Python 3.11+ and uses OpenCV 5.0.0.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

python -m simulation.synthetic_demo
python -m simulation.benchmark --episodes 200
pytest -q
```

The demo writes a visualization under `outputs/visualizations/`. The
benchmark is explicitly synthetic: it tests planning logic and provides a
baseline, not a claim about real-robot performance.

## Evaluation strategy

The system must beat meaningful baselines, not just produce a convincing demo.

1. Direct retargeted imitation without future planning.
2. Candidate planning with the geometric baseline.
3. Candidate planning with the IRASim adapter.
4. Optional learned state/action representation versus the OpenCV baseline.
5. Open-loop execution versus closed-loop visual verification/recovery.

Primary metrics: task success rate, collision rate, final target distance,
planning latency, recovery success, and calibration between predicted and
observed success.

See:
- `docs/evaluation_protocol.md`
- `docs/research_integration.md`
- `docs/aws_architecture.md`

## Repository layout

```
configs/                 experiment configuration
data/                    demonstrations and processed data
docs/                    architecture and evaluation protocol
simulation/              reproducible simulator/synthetic demos
src/
  cloud/                 AWS rollout-service boundary
  core/                  shared typed schemas
  vision/                OpenCV perception + geometry
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

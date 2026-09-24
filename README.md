# Human Motion → Robot Prediction

A computer-vision and robotics project that learns a human action from video, converts the observed motion into a robot-compatible motion representation, predicts what will happen if the robot executes that motion, and then selects/refines the safest successful motion.

## Core pipeline

Video of a human
→ visual understanding
→ human motion representation
→ robot motion mapping
→ future-outcome prediction
→ motion selection/refinement
→ robot/simulation execution

## Project goals

1. Understand actions from ordinary human videos.
2. Represent the motion in a compact form that can be transferred to a robot.
3. Predict the visual/physical outcome before execution.
4. Reject or modify motions that are likely to fail.
5. Demonstrate the complete loop on a manipulation task such as grasping a cup.

## Architecture

```
                    INPUT
              Human demonstration
                     │
                     ▼
        ┌──────────────────────────┐
        │  01_vision_understanding │
        │  frames / objects / pose │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │  02_motion_representation│
        │   human motion → compact │
        │        representation    │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │  03_robot_mapping        │
        │ human motion → robot     │
        │ compatible trajectory    │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │  04_future_prediction    │
        │ "what happens if robot   │
        │  performs this motion?"  │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │  05_motion_selection     │
        │ compare candidate        │
        │ motions / refine motion  │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │  06_execution            │
        │ simulation / robot       │
        └──────────────────────────┘
```

## Repository structure

```
cv-competition/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── demonstrations/
├── notebooks/
├── src/
│   ├── vision/
│   │   ├── video_loader.py
│   │   ├── object_tracking.py
│   │   └── human_motion.py
│   ├── motion/
│   │   ├── representation.py
│   │   └── normalization.py
│   ├── robot/
│   │   ├── mapping.py
│   │   └── trajectory.py
│   ├── prediction/
│   │   ├── future_model.py
│   │   └── rollout.py
│   ├── selection/
│   │   ├── candidate_scoring.py
│   │   └── refinement.py
│   └── pipeline.py
├── simulation/
├── tests/
└── outputs/
    ├── visualizations/
    └── predictions/
```

## First demonstration

The first end-to-end experiment should use a simple grasping task:

**human video → understand reaching/grasping motion → map it to a robot → predict the result → choose/refine the motion → execute in simulation.**

The initial version should focus on proving the pipeline rather than building a general-purpose robot system.

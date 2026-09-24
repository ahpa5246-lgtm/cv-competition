# Closed-loop visual agent

The project now distinguishes **predicted success** from **observed success**.

A planner may predict that a trajectory reaches the goal, but the environment
can change after planning. Therefore the agent uses the following loop:

```
Observe with OpenCV
      ↓
Plan candidate actions
      ↓
Imagine outcomes
      ↓
Select action
      ↓
Execute
      ↓
Observe again with OpenCV
      ↓
Visually verify
   ↙        ↘
success    failure / changed scene
 stop          ↓
             replan
```

## Recovery demo

`python -m simulation.closed_loop_demo`

The initial frame contains no obstacle, so the first plan is valid according to
the observed state. Immediately before the first execution, the simulator
introduces an obstacle. Execution stops on collision.

The second camera observation sees the new blue obstacle using OpenCV HSV
segmentation. The agent replans from its actual current position, evaluates
candidate trajectories against the newly observed obstacle, selects a route
around it, executes, and only stops when the post-action image visually
confirms the goal.

This demo validates agent orchestration, not real-world robot physics.

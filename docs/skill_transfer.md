# Human demonstration skill transfer

The project must not reduce a human video to only "there is a target."

The baseline now preserves the **shape of the demonstrated motion**.

## Representation

The demonstrated path is expressed in a local coordinate system defined by:

- the demonstrated start→goal axis;
- longitudinal progress along that axis;
- lateral deviation from that axis.

Those normalized coordinates are reconstructed around the robot's current
start→goal axis. This preserves approach geometry while adapting translation,
rotation and scale to a different scene.

## Planning behavior

The transferred human path becomes an explicit candidate:

`human-demo`

It competes with generated alternatives under the same future-model scoring.

- If the demonstrated strategy is valid in the robot scene, the planner gives
  it a small preference because it carries human information.
- If the transferred path is predicted to collide or fail, collision/risk
  penalties dominate and another candidate is selected.

This is important: the robot **learns from** the human demonstration without
being forced to **blindly imitate** it.

## Future StaMo integration

The geometric path representation is a transparent baseline. A StaMo-derived
compact state/action representation can later replace or augment it while
keeping the same experiment:

1. learn from video;
2. transfer to new embodiment/scene;
3. compare the learned proposal with alternatives;
4. use the world model to accept, reject or refine it.

The benchmark should compare transfer success and robustness rather than merely
whether inference runs.

# AWS architecture

AWS should carry a meaningful workload, not be decorative storage.

## Proposed competition deployment

**Edge / local**
1. Camera input.
2. OpenCV 5 frame processing, tracking/detection, optical flow and geometry.
3. Task-centric state construction.
4. Candidate trajectory generation.
5. Final visual verification after action.

**AWS**
1. Receive compact current-scene state + candidate robot trajectories.
2. Run expensive future/world-model rollouts (IRASim-class inference when
   integrated).
3. Evaluate rollout outcomes and uncertainty.
4. Return structured candidate outcomes to the local selector.
5. Persist experiment evidence/metrics for reproducibility.

This design keeps latency-sensitive perception and privacy-sensitive raw video
local while moving compute-heavy counterfactual rollout generation to cloud
compute.

## Repository boundary

`src/cloud/aws_rollout_client.py` implements the client-side WorldModel
contract. The deployed endpoint is intentionally not claimed as complete until
the infrastructure and inference image exist.

## Validation requirement

The final report should measure:
- local perception latency;
- network round-trip;
- AWS rollout latency per candidate/batch;
- full decision latency;
- success/collision change from enabling cloud world-model planning.

If AWS does not improve or materially enable the system, redesign its role
rather than presenting it as a checkbox.

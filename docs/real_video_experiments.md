# Real-video experiment harness

The repository includes a repeatable bridge from synthetic validation to real
recordings.

## Inputs

Each experiment uses:

1. a human demonstration video;
2. a robot-scene image/frame;
3. an experiment YAML containing HSV perception ranges and planning settings.

The harness does not silently alter model settings. The YAML should be kept
with competition evidence so results are reproducible.

## Run

```bash
python -m src.experiment.cli --config configs/real_video.example.yaml
```

## Outputs

Each run creates:

- `report.json`: software versions, input paths, learned intent, observed robot
  scene, transferred human path, candidate ranking and final decision;
- `overlay.png`: robot scene with obstacles, transferred human proposal and
  selected path.

The report explicitly records whether the selected motion came from
`human-demonstration-transfer` or `generated-alternative`.

## First recording protocol

For the initial real-video baseline, use a fixed overhead/oblique camera and
high-contrast markers:

- green marker on the demonstrator hand/tool and simulated/robot effector;
- red target marker/object;
- blue obstacle blocks.

This deliberately simple OpenCV baseline is useful before introducing learned
detection because it isolates motion-transfer and planning failures from
perception failures.

Once the end-to-end pipeline is reliable, replace HSV perception with OpenCV
DNN detection/segmentation and rerun the same experiment definitions.

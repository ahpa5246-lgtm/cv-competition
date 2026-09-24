"""End-to-end project pipeline.

Implementation will connect:
1. visual understanding
2. human motion representation
3. robot motion mapping
4. future-outcome prediction
5. motion selection/refinement
6. execution
"""

def run_pipeline(video_path: str):
    raise NotImplementedError("Pipeline components will be implemented incrementally.")


if __name__ == "__main__":
    run_pipeline("data/demonstrations/grasp_cup.mp4")

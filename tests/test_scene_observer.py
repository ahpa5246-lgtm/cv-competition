import cv2
import numpy as np

from src.agent.closed_loop import (
    DEFAULT_EFFECTOR_HSV,
    DEFAULT_OBSTACLE_HSV,
    DEFAULT_TARGET_HSV,
)
from src.vision.scene_observer import observe_scene


def test_scene_observer_recovers_effector_target_and_obstacle():
    frame = np.full((200, 300, 3), 245, dtype=np.uint8)
    cv2.circle(frame, (30, 160), 8, (0, 255, 0), -1)
    cv2.circle(frame, (260, 35), 15, (0, 0, 255), -1)
    cv2.rectangle(frame, (120, 80), (180, 130), (255, 0, 0), -1)

    scene = observe_scene(
        frame,
        effector_hsv=DEFAULT_EFFECTOR_HSV,
        target_hsv=DEFAULT_TARGET_HSV,
        obstacle_hsv=DEFAULT_OBSTACLE_HSV,
    )

    assert scene.hand_xy is not None
    assert scene.target_xy is not None
    assert len(scene.obstacle_boxes) == 1
    assert abs(scene.hand_xy[0] - 0.10) < 0.02
    assert abs(scene.target_xy[0] - (260 / 300)) < 0.02

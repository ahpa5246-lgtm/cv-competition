import cv2
import numpy as np

from src.vision.object_tracking import largest_region_centroid


def test_hsv_centroid_detection():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(frame, (80, 60), 12, (0, 255, 0), -1)

    point = largest_region_centroid(frame, (45, 100, 80), (85, 255, 255))

    assert point is not None
    assert abs(point[0] - 80) < 2
    assert abs(point[1] - 60) < 2

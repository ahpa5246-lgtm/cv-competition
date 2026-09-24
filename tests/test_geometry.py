import numpy as np

from src.vision.geometry import estimate_homography, transform_points


def test_homography_maps_image_corners_to_workspace():
    image = [(10, 10), (110, 10), (110, 60), (10, 60)]
    world = [(0, 0), (1, 0), (1, 1), (0, 1)]

    h = estimate_homography(image, world)
    mapped = transform_points([(60, 35)], h)

    assert np.allclose(mapped[0], (0.5, 0.5), atol=1e-2)

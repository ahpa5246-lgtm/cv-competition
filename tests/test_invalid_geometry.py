import numpy as np
import pytest
from src.vision.geometry import estimate_homography, transform_points
@pytest.mark.parametrize('points',[[(0.,0.),(1.,0.),(1.,1.),(float('nan'),1.)],[(0.,0.),(1.,0.),(1.,1.),(float('inf'),1.)]])
def test_homography_rejects_nonfinite_calibration_points(points):
 with pytest.raises(ValueError):estimate_homography(points,[(0.,0.),(1.,0.),(1.,1.),(0.,1.)])
def test_transform_rejects_invalid_calibration_matrix():
 with pytest.raises(ValueError):transform_points([(1.,1.)],np.full((3,3),float('nan')))

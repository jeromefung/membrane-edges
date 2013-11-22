'''
Implement a Canny edge detection scheme.

Idea is to find edges of fluorescene images of labeled M13K07
wetting the edge of a dominantly fdY21M membrane.

Heavily based on the following:
http://code.google.com/p/python-for-matlab-users/source/browse/Examples/SciPy-Canny.py?r=65c99a42f0a901ba8aec66fd9d392a6b495c889c

http://pythongeek.blogspot.com/2012/06/canny-edge-detection.html

An alternative scheme is Canny-Deriche detection. 
Deriche uses a different kernel for jointly smoothing and taking the 
gradient, which can be parallelized.

However, Sobel operators already implemented in scipy.ndimage, so 
Canny is faster to try.
'''

import scipy.ndimage as ndimage
import numpy as np

from numpy import pi


def non_max_edge_supp(mag, angle):
    # bin orientations into 4 directions
    # 0: E-W, 1: NE-SW, 2: N-S, 3: NW-SE
    angle_bins = ((angle + pi) * 4 / pi + 0.5).astype('int') % 4

    mask = np.zeros(mag.shape, dtype = 'bool')
    mask[1:-1, 1:-1] = True
    edge_map = np.zeros(mag.shape, dtype = 'bool')
    offsets = ((1, 0), (1, 1), (0, 1), (-1, 1))

    for direction, (di, dj) in zip(range(4), offsets):
        # where angle_bins in one of the bins, except for edges
        cand_idx = np.nonzero(np.logical_and(angle_bins == direction, mask))
        # check for local max
        for i, j in zip(*cand_idx):
            if mag[i, j] > mag[i + di, j + dj] and \
                    mag[i, j] > mag[i - di, j - dj]:
                edge_map[i, j] = True
    return edge_map

def detect_canny(img, sigma = 1.0, low_thr = 50, high_thr = 100):
    # gaussian smoothing
    img_smoothed = ndimage.filters.gaussian_filter(img, sigma)
    # gradient
    grad_x = ndimage.filters.sobel(img_smoothed, axis = 0)
    grad_y = ndimage.filters.sobel(img_smoothed, axis = 1)
    grad_mag = np.hypot(grad_x, grad_y)
    grad_angle = np.arctan2(grad_y, grad_x)

    # non-maximal edge suppression
    edges_suppresed = non_max_edge_supp(grad_mag, grad_angle)

    # thresholding with hysteresis
    gt_lthr = np.logical_and(edges_suppresed, grad_mag > low_thr)
    # Does magic to detect connectivity
    # use 8-fold neighbor kernel instead of default
    labels, num_labels = ndimage.measurements.label(gt_lthr,
                                                    np.ones((3,3)))
    for i in range(num_labels):
        if max(grad_mag[labels == i]) < high_thr:
            gt_lthr[labels == i] = False

    return gt_lthr



            

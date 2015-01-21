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
from collections import deque

def grad_mag_fd(image):
    finite_diff_kernel = np.array([1., 0., -1.])
    grad_x = ndimage.convolve1d(image, finite_diff_kernel, axis = 0)
    grad_y = ndimage.convolve1d(image, finite_diff_kernel, axis = 1)
    # factor of 4 keeps magnitude comparable to Sobel
    return 4. * np.hypot(grad_x, grad_y)

def grad_mag_sobel(image):
    grad_x = ndimage.filters.sobel(image, axis = 0)
    grad_y = ndimage.filters.sobel(image, axis = 1)
    return np.hypot(grad_x, grad_y)

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


def non_min_edge_supp(mag, angle):
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
            if mag[i, j] < mag[i + di, j + dj] and \
                    mag[i, j] < mag[i - di, j - dj]:
                edge_map[i, j] = True
    return edge_map


def detect_canny(img, sigma = 1.0, low_thr = 50, high_thr = 100, 
                 return_gradient = False):
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

    if return_gradient:
        return gt_lthr, grad_mag
    else:
        return gt_lthr


def detect_ridge(img, sigma = 1.0, int_thr = 1000, 
                 return_gradient = False):
    # gaussian smoothing
    img_smoothed = ndimage.filters.gaussian_filter(img, sigma)
    # gradient
    grad_x = ndimage.filters.sobel(img_smoothed, axis = 0)
    grad_y = ndimage.filters.sobel(img_smoothed, axis = 1)
    grad_mag = np.hypot(grad_x, grad_y)
    grad_angle = np.arctan2(grad_y, grad_x)

    # non-minimal edge suppression
    edges_suppressed = non_min_edge_supp(grad_mag, grad_angle)    

    # original image thresholding
    ridge = np.logical_and(edges_suppressed, img > int_thr)
  
    if return_gradient:
        return ridge, grad_mag
    else:
        return ridge        


# now need edge cleanup function
def ridge_order(ridge_img, img, starting_point = None):
    '''
    Returns: list of tuples
    Use intensity img as tiebreaker in corner case
    '''
    # get list of tuples (coordinates)
    xcoords, ycoords = np.where(ridge_img)
    indices = np.lexsort((ycoords, xcoords)) # sort on x, then y
    ridge_pnts = deque([(xcoords[i], ycoords[i]) for i in indices])

    if starting_point == None:
        starting_point = ridge_pnts[0]

    # initialize
    output = [starting_point]
    ridge_pnts.remove(starting_point)

    # main loop
    while len(ridge_pnts) > 0:
        active_pnt = output[-1]
        # 8 - fold neighborhood
        neighborhood = [(active_pnt[0] + 1, active_pnt[1]),
                        (active_pnt[0] - 1, active_pnt[1]),
                        (active_pnt[0] + 1, active_pnt[1] + 1),
                        (active_pnt[0] - 1, active_pnt[1] + 1),
                        (active_pnt[0], active_pnt[1] + 1),
                        (active_pnt[0] + 1, active_pnt[1] - 1),
                        (active_pnt[0] - 1, active_pnt[1] - 1),
                        (active_pnt[0], active_pnt[1] - 1)]
        check = np.array([pt in ridge_pnts for pt in neighborhood])
        
        if len(np.where(check)[0]) == 0:
            # no remaining ridge points in 8-neighborhood
            # calculate distance to all other points
            remaining_x = np.array([i[0] for i in ridge_pnts])
            remaining_y = np.array([i[1] for i in ridge_pnts])
            distances = np.hypot(active_pnt[0] - remaining_x,
                                 active_pnt[1] - remaining_y)
            neighbor = ridge_pnts[np.argmin(distances)]
            output.append(neighbor)
            ridge_pnts.remove(neighbor)
        elif len(np.where(check)[0]) == 1:
            # one single pt
            neighbor = neighborhood[np.where(check)[0][0]]
            output.append(neighbor)
            ridge_pnts.remove(neighbor)
        else: 
            # find closest point in the neighborhood
            candidates = []
            for pt, val in zip(neighborhood, check):
                if val:
                    candidates.append(pt)

            neighbor_x = np.array([i[0] for i in candidates])
            neighbor_y = np.array([i[1] for i in candidates])
            distances = np.hypot(active_pnt[0] - neighbor_x,
                                 active_pnt[1] - neighbor_y)
            nearest = np.where(distances == distances.min())[0]
            if len(nearest) == 1:
                neighbor = candidates[nearest[0]]
                output.append(neighbor)
                ridge_pnts.remove(neighbor)
            elif len(nearest) == 2:
                # catch corner case of multiple pixel thick segment
                # go with max 
                if img[candidates[nearest[0]]] > img[candidates[nearest[1]]]:
                    # throw out low point
                    ridge_pnts.remove(candidates[nearest[1]])
                    output.append(candidates[nearest[0]])
                    ridge_pnts.remove(candidates[nearest[0]])
                else:
                    ridge_pnts.remove(candidates[nearest[0]])
                    output.append(candidates[nearest[1]])
                    ridge_pnts.remove(candidates[nearest[1]])
                print 'Ridge ordering: corner case'
            else:
                # ambiguous situation
                raise RuntimeError('Ridge point has multiple next neighbors')

    return output


    



        

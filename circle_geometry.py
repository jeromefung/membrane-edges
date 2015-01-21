
import numpy as np
import itertools

from numpy import array, sqrt
from numpy.linalg import det


def circle_three_pts(p0, p1, p2):
    '''
    Calculate parameters of a circle given 3 noncollinear points:

    http://mathworld.wolfram.com/Circle.html
    '''
    a = det(array([[p0[0], p0[1], 1.],
                   [p1[0], p1[1], 1.],
                   [p2[0], p2[1], 1.]]))
    d = -det(array([[p0[0]**2 + p0[1]**2, p0[1], 1.],
                    [p1[0]**2 + p1[1]**2, p1[1], 1.],
                    [p2[0]**2 + p2[1]**2, p2[1], 1.]]))
    e = det(array([[p0[0]**2 + p0[1]**2, p0[0], 1.],
                    [p1[0]**2 + p1[1]**2, p1[0], 1.],
                    [p2[0]**2 + p2[1]**2, p2[0], 1.]]))
    f = -det(array([[p0[0]**2 + p0[1]**2, p0[0], p0[1]],
                    [p1[0]**2 + p1[1]**2, p1[0], p1[1]],
                    [p2[0]**2 + p2[1]**2, p2[0], p2[1]]]))

    r = sqrt((d**2 + e**2)/(4. * a**2) - f/a)
    x0 = -d/(2.*a)
    y0 = -e/(2.*a)

    return x0, y0, r


def estimate_circle_parameters(pts):
    '''
    pts: ndarray with n_pts rows
    pick triplets and get median
    '''

    n_pts = pts.shape[0]
    n_permutations = n_pts * (n_pts - 1) * (n_pts - 2)

    x0 = np.zeros(n_permutations)
    y0 = np.zeros(n_permutations)
    r0 = np.zeros(n_permutations)

    for perm, ctr in zip(itertools.permutations(pts, 3), 
                         np.arange(n_permutations)):
        x0[ctr], y0[ctr], r0[ctr] = circle_three_pts(*perm)

    return np.median(x0), np.median(y0), np.median(r0)


'''
Follow method in

In-Kwon Lee, "Curve reconstruction from unorganized points,"
Computer Aided Geometric Design 17 (2000), 161-177.
'''

import numpy as np

from numpy.polynomial import polynomial 

def thin_and_order(xpts, ypts, weighting_H = None, spacing = 3.):
    if weighting_H == None:
        weighting_H = len(xpts) / 5.

    # choose starting point at midpoint #at random
    #start_index = np.random.random_integers(0, len(xpts) - 1)
    start_index = np.floor(len(xpts)/2.)

    def main_loop(x0, y0):
        # get local slope at this point
        distances = calc_distances(xpts, ypts, x0, y0)
        coeffs = weighted_local_regression(xpts, ypts, distances,
                                           weighting_H)

        # find nearby points 
        nearby_xs = xpts[distances < spacing]
        nearby_ys = ypts[distances < spacing]
        nearby_dists = distances[distances < spacing]

        # L vector: 1 i + p1 j
        dot_prods = (nearby_xs - x0) + (nearby_ys - y0) * coeffs[1]
        group1_xs = nearby_xs[dot_prods > 0.]
        group2_xs = nearby_xs[dot_prods < 0.]
        group1_ys = nearby_ys[dot_prods > 0.]
        group2_ys = nearby_ys[dot_prods < 0.]

        distances_group1 = nearby_dists[dot_prods > 0.]
        distances_group2 = nearby_dists[dot_prods < 0.]

        group1_out = np.array([group1_xs[distances_group1.argmax()],
                               group1_ys[distances_group1.argmax()]])
        group2_out = np.array([group2_xs[distances_group2.argmax()],
                               group2_ys[distances_group2.argmax()]])
        return group1_out, group2_out

    # do main loop on first point
    p1a, p1b = main_loop(xpts[start_index], ypts[start_index])

    # right list
    right_pts = [np.array([xpts[start_index], ypts[start_index]]),
                 p1b]
    
    while len(right_pts) < len(xpts):
        try:
            working_pt = right_pts[-1]
            prev_working_pt = right_pts[-2]
            pnext_a, pnext_b = main_loop(*working_pt)
            if (np.hypot(pnext_a[0] - prev_working_pt[0], 
                         pnext_a[1] - prev_working_pt[1]) >= 
                np.hypot(pnext_b[0] - prev_working_pt[0],
                         pnext_b[1] - prev_working_pt[1])):
                right_pts.append(pnext_a)
            else:
                right_pts.append(pnext_b)
        except (IndexError, ValueError): # main loop ran out of points
            break

    # left list
    left_pts = [np.array([xpts[start_index], ypts[start_index]]),
                 p1a]
    while len(left_pts) < len(xpts):
        try:
            working_pt = left_pts[-1]
            prev_working_pt = left_pts[-2]
            pnext_a, pnext_b = main_loop(*working_pt)
            #print pnext_a, pnext_b
            if (np.hypot(pnext_a[0] - prev_working_pt[0], 
                         pnext_a[1] - prev_working_pt[1]) >= 
                np.hypot(pnext_b[0] - prev_working_pt[0],
                         pnext_b[1] - prev_working_pt[1])):
                left_pts.append(pnext_a)
            else:
                left_pts.append(pnext_b)
        except (IndexError, ValueError): # main loop ran out of points
            break

    #print left_pts
    #print right_pts
    return np.array(left_pts[::-1][:-1] + right_pts)
                    


def calc_distances(xpts, ypts, x0, y0):
    return np.hypot(xpts - x0, ypts - y0)


def weighted_local_regression(xpts, ypts, dists_from_p0, weighting_H):
    weights = np.exp(-dists_from_p0**2 / (0.5 * weighting_H**2)) 
    # weights will be squared, hence factor of 0.5
    return polynomial.polyfit(xpts, ypts, deg = 1, w = weights)
    

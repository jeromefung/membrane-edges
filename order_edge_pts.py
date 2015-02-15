
import numpy as np
import edge_detector

from numpy import pi

def order_edge_polar(edge, membrane_rad, angle_range, 
                     dist_from_avg = 30.):
    '''
    edge: boolean array
    membrane_rad: guess for radius
    dist_from_avg: reject edge points further than this from avg radius
    '''
    xpts, ypts = np.where(edge)
    circ_params_rough = edge_detector.fit_circ_to_edge(edge,
                                                       [xpts.mean(),
                                                        ypts.mean()],
                                                       membrane_rad)
    xgrid, ygrid = np.mgrid[0:edge.shape[0], 0:edge.shape[1]]
    radii = np.hypot(xgrid - circ_params_rough[0],
                     ygrid - circ_params_rough[1])
    #print radii.shape
    better_edge = (edge * 
                   (radii > (circ_params_rough[2] - dist_from_avg)) * 
                   (radii < (circ_params_rough[2] + dist_from_avg)))
    xpts, ypts = np.where(better_edge)
    circ_params = edge_detector.fit_circ_to_edge(better_edge,
                                                 circ_params_rough[0:2],
                                                 circ_params_rough[2])
    # order points in polar coordinates
    phis = np.array([np.arctan2(y - circ_params[1], x - circ_params[0])
                     for (x, y) in zip(xpts, ypts)])
    #raise Exception
    # wrap negative angles to positive (0 - 2pi rather than -pi - pi)
    if angle_range[0] >= 0. and angle_range[1] >= np.pi:
        phis[phis < 0] += 2 * np.pi

    ordering = np.argsort(phis)
    xpts = xpts[ordering]
    ypts = ypts[ordering]
    phis.sort()
    
    start_ind = np.where(phis >= angle_range[0])[0][0]
    stop_ind = np.where(phis <= angle_range[1])[0][-1]
    
    return np.vstack((xpts[start_ind : stop_ind + 1],
                      ypts[start_ind : stop_ind + 1])).transpose()


def angle_wrapper(phis, epsilon = 0.02):
    '''
    arctan's output goes from -pi to pi.  At pi there is a discontinuity.
    If you have a membrane where there are phis crossing this continuity,
    you can get numerics (tangent angles oscillate wildly).

    Check for a "pi singularity" and treat angles appropriately.
    '''
    # check pi singularity
    if (phis > pi - epsilon).any() and (phis < -pi + epsilon).any():
        # yes, check if there's also data crossing 0:
        if (np.abs(phis) < epsilon).any():
            # find "start angle", send angles btw it and pi back by 2pi.
            phis_quads12 = np.sort(phis[(phis >= 0) * (phis <= pi)])
            differences = roll(phis_quads12, -1) - phis_quads12
            start_angle = phis_quads12[differences.argmax() + 1]
            phis[phis > start_angle] -= 2. * pi
        else:
            # wrap angles to 0-2pi, add 2pi to negative angles
            phis[phis < 0.] += 2. * pi

    return phis


def get_tangents(refined_spl):
    # uniformly spaced
    if refined_spl.uniform_u is None:
        refined_spl.uniform_arclength_u()

    # x, y, ui, tx, ty, tan
    length = len(refined_spl.uniform_u)
    output = np.zeros((length, 6))
    
    for ui, ctr in zip(refined_spl.uniform_u, range(length)):
        output[ctr, 0:2] = refined_spl.evaluate(ui)
        output[ctr, 2] = ui
        xder, yder = refined_spl.derivative(ui)
        output[ctr, 3:5] = np.array([xder, yder])
        output[ctr, 5]  = np.arctan2(yder, xder)

    return output


def sort_layer_edge_pts(edge, membrane_rad_guess, min_gap_size = 4.):
    '''
    Take edge image Canny-detected for a wetting edge. Sort into
    membrane edge vs. layer inner edge.

    Method: fit a circle to all the edge points. This doesn't do very
    well at getting a circle where all membrane edge points are outside
    and all inner edge points are inside. However, calculating distances
    from fitted center does provide a means of sorting.
    '''

    xpts, ypts = np.where(edge)
    # fit circle
    circ_params = edge_detector.fit_circ_to_edge(edge,
                                                 [xpts.mean(), ypts.mean()],
                                                 membrane_rad_guess)
    dists_from_center = np.hypot(xpts - circ_params[0], ypts - circ_params[1])
    sorted_dists = np.sort(dists_from_center)

    # find gaps
    gaps = (np.roll(sorted_dists, -1) - sorted_dists)[:-1]
    
    if gaps.max() > min_gap_size:
        split_point = sorted_dists[gaps.argmax() : gaps.argmax() + 2].mean()
        outer_x = xpts[dists_from_center > split_point]
        outer_y = ypts[dists_from_center > split_point]
        inner_x = xpts[dists_from_center < split_point]
        inner_y = ypts[dists_from_center < split_point]
        return np.array([outer_x, outer_y]), np.array([inner_x, inner_y])
    else:
        raise RuntimeError('Biggest gap in distance array too small')
                                                 

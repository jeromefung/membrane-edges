import numpy as np
from scipy.stats import linregress

def fix_2pi(tangent_arr):
    working_arr = tangent_arr.copy()
    n_angles = tangent_arr.shape[0]
    angles = working_arr[:,-1]

    # fit a line
    xs = np.arange(n_angles)
    params = linregress(xs, angles)
    bf_angles = xs * params[0] + params[1]

    indices_to_check = np.where(np.abs(bf_angles - angles) > 5.)[0]
    
    for idx in indices_to_check:
        # pick the one closest to best-fit line
        candidate_angles = np.array([angles[idx],
                                     angles[idx] + 2. * np.pi,
                                     angles[idx] - 2. * np.pi])
        distances_from_line = np.abs(candidate_angles - bf_angles[idx])
        angles[idx] = candidate_angles[distances_from_line.argmin()]

    return working_arr


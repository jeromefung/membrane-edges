'''
Code to fit a linear cut of derivative image
'''

import numpy as np
import scipy.ndimage
from numpy import exp
from scipy.optimize import curve_fit



def single_gaussian(x, x0, A, sigma):
    return A * exp(-(x - x0)**2 / (2. * sigma**2))


def fit_one_gaussian(x, y, r0, A0, sigma_0):
    p0 = np.array([r0, A0, sigma_0])
    params, cov = curve_fit(single_gaussian, x, y, p0 = p0)
    return params


def fit_2_gaussians(r, mag, r0, sigma, window_width):
    '''
    Inputs:
    r: 1d array of radial positions
    mag: 1d array of intensities corresponding to r
    r0: 1d array (2), guesses for centers of 2 gaussians
    window_width: float, units of r, fit gaussians +/- window_width from r0

    Returns:
    peaks: 1d array (2), detected peak positions in radial units

    Notes:
    This will only work well when fitting near the tops of two peaks,
    that are assumed to be non-overlapping
    '''
    def extract_region(ctr):
        index_arr = np.logical_and(r > ctr - window_width, 
                                   r < ctr + window_width)
        return r[index_arr], mag[index_arr]

    output = np.array([])

    for ctr in r0:
        r_trunc, mag_trunc = extract_region(ctr)
        output = np.append(output, fit_one_gaussian(r_trunc, mag_trunc, ctr,
                                                    mag_trunc.max(), sigma))
    return output.reshape((2,3))

    
def auto_detect_2pks(x, y, thr = 0.5):
    '''
    Simple peak detection by looking for connected regions above threshold.
    Return position of local max in regions with 2 biggest maxima.
    '''
    labels, n_labels = scipy.ndimage.measurements.label(y > thr * y.max())
    max_in_regions = np.array([y[labels == i].max() for i in 
                               np.arange(n_labels) + 1]) # labels count from 1
    max_to_return = np.sort(max_in_regions)[-2:] # keep 2 highest
    maxima_pos = np.array([np.where(y == maxm)[0][0] for 
                           maxm in max_to_return])
    maxima_pos.sort() # smaller x first
    return x[maxima_pos], y[maxima_pos]

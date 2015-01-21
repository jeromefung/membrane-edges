'''
Calculate Fourier coefficients of a 1D function.

This code improves over Ed Barry's by using Simpson's rule
to perform the numerical integration to calculate Fourier coefficients,
instead of what amounts to using a rectangular Riemann sum. 

Also, take care to consider Nyquist.
'''

import numpy as np
from numpy import arange, sin, cos, pi, floor
from scipy.integrate import simps

def coeffs_n(y, n, x = None):
    if x == None:
        x = arange(len(y))
    L = x[-1] - x[0]
    cos_terms = cos(n * 2. * pi * x / L)
    sin_terms = sin(n * 2. * pi * x / L)
    # off by factor of 2 for a_0
    a_n = 2. / L * simps(y * cos_terms, x)
    b_n = 2. / L * simps(y * sin_terms, x)
    return a_n, b_n


def fourier_coeffs(y, x = None, n_max = None):
    '''
    output from n = 0
    '''
    if x == None:
        x = arange(len(y))
    if n_max == None:
        n_max = floor(len(y)/2)
    coeffs = np.array([coeffs_n(y, n, x) for n in arange(n_max + 1)])
    a_ns = coeffs[:,0]
    # fix factor of 2 in a_0
    a_ns[0] *= 0.5
    b_ns = coeffs[:,1]
    return a_ns, b_ns


'''
Do mode analysis on cluster edges, r(phi). Need to compute Fourier
integrals over phi.

2 possibilities:
    Use quadrature on known points evenly spaced in arclength, but not
    constant in phi

    Calculate Fourier integrals directly from spline representation
'''

import numpy as np
import scipy.integrate as integrate

from thread_flucts import spline_integrate
from numpy import sin, cos, arctan2, pi

def coeffs_by_quadrature(points, center, r_0, kmax):
    '''
    points: in cartesian coordinates
    '''
    shifted_x = points[0] - center[0]
    shifted_y = points[1] - center[1]

    r = np.hypot(shifted_x, shifted_y)
    phi = arctan2(shifted_y, shifted_x)

    sorted_r = r[np.argsort(phi)]
    sorted_phi = np.sort(phi)
    
    # add points so that endpoints are really +/- pi. continue integrand.
    sorted_phi = np.concatenate((np.array([-pi]),
                                 sorted_phi,
                                 np.array([pi])))
    sorted_r = np.concatenate((np.array([sorted_r[0]]),
                               sorted_r,
                               np.array([sorted_r[-1]])))

    aks = np.zeros(kmax + 1)
    bks = np.zeros(kmax + 1)

    # a_0
    aks[0] = integrate.cumtrapz(sorted_r/r_0, sorted_phi)[-1]/(2. * pi) - 1.

    for k in np.arange(1, kmax + 1):
        aks[k] = 1./pi * integrate.cumtrapz(sorted_r/r_0 * cos(k * sorted_phi), 
                                            sorted_phi)[-1]
        bks[k] = 1./pi * integrate.cumtrapz(sorted_r/r_0 * sin(k * sorted_phi),
                                            sorted_phi)[-1]

    return aks, bks


def coeffs_from_splines(spl, center, r_0, kmax):
    '''
    spl: instance of ParametricSpline2D

    Do integrals over control variable u
    '''

    def coords(ui):
        x, y = spl.evaluate(ui)
        phi = arctan2(y - center[1], x - center[0])
        r = np.hypot(x - center[0], y - center[1])
        return r, phi, x - center[0], y - center[1]

    def make_integrand(k, mode):
        '''
        mode = 1: cos
        mode = 2: sin
        '''
        def integrand_c(ui):
            r, phi, x, y = coords(ui)
            der_x, der_y = spl.derivative(ui)
            if mode == 1:
                prefactor = r / r_0 * cos(k * phi)
            elif mode == 2:
                prefactor = r / r_0 * sin(k * phi)
            du = 1. / (x * (1 + (y / x)**2)) * (der_y - y / x * der_x)
            return prefactor * du

        return integrand_c
    
    aks = np.zeros(kmax + 1)
    bks = np.zeros(kmax + 1)

    # a_0
    aks[0] = spline_integrate(make_integrand(0, 1), 0., 1., 
                              spl.tck[0]) / (2. * pi) - 1

    for k in np.arange(1, kmax + 1):
        aks[k] = 1. / pi * spline_integrate(make_integrand(k, 1), 0., 1., 
                                            spl.tck[0])
        bks[k] = 1. / pi * spline_integrate(make_integrand(k, 2), 0., 1.,
                                            spl.tck[0])

    return aks, bks

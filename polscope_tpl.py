import numpy as np
from numpy import sin, exp, arctan, tan, array
from scipy.integrate import quad, Inf

def retardance(x, C, theta_0, lambda_t):
    if x >= 0:
        return C * sin(2. * arctan(tan(theta_0/2.) * exp(-x/lambda_t)))**2
    else:
        return 0

def airy_gauss(x, x0, w):
    return exp(-(x - x0)**2 / w**2)

def fitting_function(x, C, theta_0, lambda_t, w, shift_dist, vshift):
    # use quadrature to do convolution
    # scipy.integrate.quad can handle infinite limits

    def integrand(xii, xi, C, theta_0, lambda_t, w):
        return retardance(xii, C, theta_0, lambda_t) * airy_gauss(xii, xi, w)

    def point_func(xi):
        # integrate from 0 since integrand is 0 for xii < 0
        return quad(integrand, 0, Inf, args = (xi, C, theta_0, lambda_t, 
                                                  w))[0]

    if np.isscalar(x):
        x = array([x])

    return array([point_func(xi - shift_dist) for xi in x]) + vshift






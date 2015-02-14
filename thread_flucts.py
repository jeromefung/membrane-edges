import numpy as np

import scipy.interpolate as interpolate
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.ndimage as ndimage

import edge_finder

def spline_integrate(integrand, u0, u1, knot_vector):
    '''
    Integrate an integrand defined from a spline representation 
    piecewise between knots.

    Assume smoothness between knots such that Gaussian quadrature
    can be applied.

    knot vector: tck[0] from splprep
    '''

    # select knots between u0 and u1
    knot_vector = np.array(knot_vector)
    knot_selection = (knot_vector > u0) * (knot_vector < u1)
    knots = np.concatenate((np.array([u0]),
                            knot_vector[knot_selection],
                            np.array([u1])))
    n_intervals = len(knots) - 1

    integral_pieces = np.array([integrate.fixed_quad(integrand,
                                                     knots[i], knots[i+1])[0]
                                for i in np.arange(n_intervals)])
    return integral_pieces.sum()


def fit_spline_to_ridge(ordered_ridge_pts, smooth):
    '''
    Inputs: 
        ordered_ridge_pts: output of canny_edges.ridge_order (list of tuples)
        smooth: smoothing factor passed to interpolate.splprep
    '''
    # extract coordinates
    xi = np.array([pt[0] for pt in ordered_ridge_pts])
    yi = np.array([pt[1] for pt in ordered_ridge_pts])

    # tck: tuple: vector of knots, spline coeffs, degree of spline
    # u: parametric variable, 0 to 1
    tck, u = interpolate.splprep([xi, yi], s = smooth)
    
    return tck


def fit_ui(val, spl, i = 0, lb = 0, ub = 1.):
    '''
    Given point x and ParametricSpline2D spl, find u such that
    spl.evaluate(u)[i] = x.
    '''
    def root_func(u):
        return spl.evaluate(u)[i] - val

    return optimize.brentq(root_func, lb, ub)


class ParametricSpline2D:
    def __init__(self, tck):
        self.tck = tck
        self.length = None
        self.uniform_u = None

    def _arclength_integrand(self, ui):
        der_x, der_y = self.derivative(ui)
        return np.hypot(der_x, der_y)

    def evaluate(self, ui):
        return interpolate.splev(ui, self.tck)

    def derivative(self, ui):
        return interpolate.splev(ui, self.tck, der = 1)

    def local_tangent(self, ui):
        tangent_vector = np.array(self.derivative(ui))
        return tangent_vector / np.hypot(*tangent_vector)

    def local_normal(self, ui):
        # take cross product with zhat
        return np.roll(self.local_tangent(ui), 1) * np.array([1., -1.])
    
    def calc_arclength(self, u_0, u_1):
        '''
        Compute arc length for parametric spline curve in 2D 
        (x(u) xhat + y(u) yhat)
        
        Return: ndarray
        '''
        if np.isscalar(u_1):
            u_1 = np.array([u_1])

        integral = np.array([spline_integrate(self._arclength_integrand, u_0, 
                                              ui, self.tck[0]) for ui in u_1])
        return integral

    def calc_total_length(self):
        self.length = self.calc_arclength(0., 1.)[0]
        return self.length

    def uniform_arclength_u(self, interval = 1.):
        def _root_function(us, u0):
            return self.calc_arclength(u0, us) - interval

        def _root_deriv(ui, dummy = None):
            '''
            Dummy argument so that derivative has same number of
            args as _root_function
            '''
            return self._arclength_integrand(ui)
            
        # calculate length
        if self.length == None:
            self.calc_total_length()
        
        guess_interval = 1./self.length
        output_u = np.array([0.])

        while output_u[-1] < 1.:
            next_u = optimize.newton(_root_function, 
                                     x0 = output_u[-1] + guess_interval,
                                     fprime = _root_deriv,
                                     args = (output_u[-1],))
            output_u = np.append(output_u, next_u)
            
        self.uniform_u = output_u[output_u < 1.]
        return self.uniform_u
        

def refine_ridge(intensity_img, spline, window = 5):
    '''
    Refine Canny'ed ridge positions to subpixel level
    '''
    # check for uniform u
    if spline.uniform_u is None:
        spline.uniform_arclength_u()

    output = np.zeros((len(spline.uniform_u), 2))

    cut_range = np.arange(-window, window + 1)

    for ui, ctr in zip(spline.uniform_u, range(len(spline.uniform_u))):
        # local normal
        normal = spline.local_normal(ui)
        x0, y0 = spline.evaluate(ui)
        cut_x = x0 + cut_range * normal[0]
        cut_y = y0 + cut_range * normal[1]
        cut = ndimage.interpolation.map_coordinates(intensity_img, 
                                                    [cut_x, cut_y], order = 1)
        # fit gaussian to cut
        gauss_params = edge_finder.fit_one_gaussian(cut_range, cut,
                                                    0.1, cut.max(), 
                                                    window / 2.)
        # convert cut coordinate back to image coords
        output[ctr] = gauss_params[0] * normal + np.array([x0, y0])
       
    return output
        

def thread_thickness(gradient_img, par_spline, cut_range, pk_sigma = 3.,
                     pk_window = 5, full_output = False, spline_range = None):
    # make a ParametricSpline2D object
    # calculate uniform_u
    if par_spline.uniform_u is None:
        par_spline.uniform_arclength_u()

    output_length = len(par_spline.uniform_u)
    output = np.zeros(output_length)

    cuts = []
    out_gaussians = []
    edges_0 = []
    edges_1 = []

    # loop over uniform_u
    for ui, ctr in zip(par_spline.uniform_u, range(output_length)):
        # check if ui is within cutoff range:
        if spline_range is not None:
            if ui > spline_range[1] or ui < spline_range[0]:
                continue

        normal = par_spline.local_normal(ui)

        # calculate cut line
        cut_distances = np.arange(-cut_range, cut_range + 1) 
        x0, y0 = par_spline.evaluate(ui)
        x_cut = x0 + normal[0] * cut_distances
        y_cut = y0 + normal[1] * cut_distances
        
        # extract points on cut line from gradient image
        cut = ndimage.interpolation.map_coordinates(gradient_img, 
                                                    [x_cut, y_cut],
                                                    order = 1)
        cuts.append(cut)
        
        # try to catch edge situations where cut misses peak 
        if len(np.where(cut == 0)[0]) > 0.35 * len(cut):
            print('Edge case')
            out_gaussians.append(np.zeros((2,3)))
            continue
                 
        # autodetect peaks
        maxima_pos, maxima = edge_finder.auto_detect_2pks(cut_distances, cut)
        # fit gaussians
        try:
            gaussians = edge_finder.fit_2_gaussians(cut_distances, 
                                                    cut, maxima_pos, 
                                                    sigma = pk_sigma, 
                                                    window_width = pk_window)
            thickness = gaussians[1, 0] - gaussians[0, 0]
            output[ctr] = thickness
            out_gaussians.append(gaussians)
            # convert cut coordinate back to image coords
            edge_0 = gaussians[0, 0] * normal + np.array([x0, y0])
            edge_1 = gaussians[1, 0] * normal + np.array([x0, y0])
            edges_0.append(edge_0)
            edges_1.append(edge_1)
        except ValueError: # 2 gaussian fit fails, stop
            break
    
    if full_output:
        # for testing, will want to save all the radial cuts
        # the gaussian fit results (check goodness of fit)
        return output, cuts, out_gaussians
    else:
        return output, np.array(edges_0), np.array(edges_1)



def spline_enclosed_area(spl):
    '''
    Given a spline with periodic boundary conditions, compute the enclosed
    area.
    '''

    def integrand(ui):
        x, y = spl.evaluate(ui)
        xprime, yprime = spl.derivative(ui)
        return 0.5 * (x*yprime - y*xprime)

    return spline_integrate(integrand, 0., 1., spl.tck[0])


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

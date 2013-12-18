'''
Given: output from Canny edge detector, filtered and/or cropped
to only have membrane edge and interface edge

We now need to:
    fit a circle to the membrane edge
    if necessary, crop so that circle is a good fit
    for every point on interface edge: find circle mapping point
    go around circle

'''
import numpy as np
import scipy.optimize
import scipy.ndimage
import canny_edges
import edge_finder
from scipy.interpolate import interp1d
from numpy import sin, cos

def circular_mask(shape, ctr, radius, inner = True):
    y, x = np.ogrid[-ctr[0]:(shape[0]-ctr[0]), -ctr[1]:(shape[1]-ctr[1])]
    if inner: # blocks everything within mask
        mask = x**2 + y**2 >= radius**2
    else: # blocks everything outside mask
        mask = x**2 + y**2 <= radius **2
    return mask


def fit_circ_to_edge(img, guess_ctr, guess_r):
    '''
    img: should have only points to fit on it
    '''
    x_tofit, y_tofit = np.where(img)
    x_tofit = np.array(x_tofit)
    y_tofit = np.array(y_tofit)

    def err_func(p):
        x0, y0, r = p
        #print r, x_tofit, y_tofit, x0, y0
        return r**2 - (x_tofit - x0)**2 - (y_tofit - y0)**2

    guess_params = [guess_ctr[0], guess_ctr[1], guess_r]
    bestfit_params, success = scipy.optimize.leastsq(err_func, guess_params)
    return bestfit_params


def interface_pol_coords(interface_img, ctr):
    xi, yi = np.where(interface_img)
    x_rel_to_center = xi - ctr[0]
    y_rel_to_center = yi - ctr[1]
    phis = np.arctan2(y_rel_to_center, x_rel_to_center)
    rs = np.hypot(x_rel_to_center, y_rel_to_center)
    # sort this on phis
    output = np.array([rs, phis])
    return output[:, output[1,:].argsort()]


def thickness_by_fitting(gradient_img, circle_ctr, phi_range, r_range,  
                         int_radius, pk_sigma = 3., pk_window = 5):
    '''
    Use interpolation and Gaussian fitting of the derivative image
    to measure interfacial thickness.

    Inputs:
    gradient_img: input (I use mag squared of Sobel derivatives)
    circle_ctr: origin for polar coordinates (center of membrane)
    phi_range: array(2): min/max azimuthal angle
    r_range: array(2), min/max radius (pixel units)
    int_radius: average radius of interface
    pk_sigma: initial guess for width of gaussian peaks
    pk_window: +/- range from max position to fit gaussian

    Outputs:
    inner_peaks, outer_peaks
    '''
    rs = np.arange(r_range[0], r_range[1])
    # Follow arclength around interface in units of 1 pixel
    dphi = 1. / int_radius 
    phis = np.arange(phi_range[0], phi_range[1], dphi)
    
    interface_r = np.zeros(len(phis))
    edge_r = np.zeros(len(phis))

    for phi, counter in zip(phis, np.arange(len(phis))):
        # Extract interpolated radial cut of gradient image at each phi
        xi = rs * cos(phi) + circle_ctr[0]
        yi = rs * sin(phi) + circle_ctr[1]
        cut = scipy.ndimage.interpolation.map_coordinates(gradient_img,
                                                          [xi, yi])
        
        # autodetect two strongest peaks
        maxima_pos, maxima = edge_finder.auto_detect_2pks(rs, cut)
        # fit gaussians
        gaussians = edge_finder.fit_2_gaussians(rs, cut, maxima_pos, 
                                                sigma = pk_sigma, 
                                                window_width = pk_window)
        gauss_ctr_radii = gaussians[:,0]
        interface_r[counter] = gauss_ctr_radii[0]
        edge_r[counter] = gauss_ctr_radii[1]
        
    interface_x = interface_r * cos(phis) + circle_ctr[0]
    interface_y = interface_r * sin(phis) + circle_ctr[1]
    edge_x = edge_r * cos(phis) + circle_ctr[0]
    edge_y = edge_r * sin(phis) + circle_ctr[1]
    thickness = edge_r - interface_r
    return thickness, np.array([interface_x, interface_y]), \
        np.array([edge_x, edge_y])
                 

def height_along_circ_interface(int_polar):
    r_avg = int_polar[0].mean()
    s_observed = r_avg * int_polar[1]

    # minus sign so that height is positive going into the membrane
    interpolator = interp1d(s_observed, -(int_polar[0] - r_avg))
    
    # s spaced in pixel units
    s_plot = np.arange(s_observed[0], s_observed[-1])

    return interpolator(s_plot)
    

def find_edges_and_height(img, sigma, canny_thresh, crop_x_range,
                          crop_y_range, phi_range, r_range, 
                          inner_mask_r, center_guess, 
                          membrane_guess_r, film_thick):
    '''
    img: target raw fluorescence image
    sigma: width of gaussian kernel for blur in Canny algorithm
    canny_thresh: [low, high] thresholds for Canny
    crop_x_range: pixel range for cropping along rows
    crop_y_range: range for cropping along cols
    phi_range: range of polar angles to extract height from (radians)
    inner_mask_r: radius of mask to block interface
    center_guess: guess for circle center fit to membrane edge (cropped units)
    membrane_guess_r: guess for membrane radius
    film_thick: approximate film thickness for masking membrane interface
    '''
    edge_image, grad_image = canny_edges.detect_canny(img, sigma = sigma, 
                                                      low_thr = canny_thresh[0],
                                                      high_thr = 
                                                      canny_thresh[1], 
                                                      return_gradient = True)
    
    def crop(img):
        return img[crop_x_range[0]:crop_x_range[1], 
                   crop_y_range[0]:crop_y_range[1]]

    # find membrane edge circle
    edge_image_cropped = crop(edge_image)
    inner_mask = circular_mask(edge_image_cropped.shape, center_guess, 
                               inner_mask_r)
    membrane_edge = np.logical_and(inner_mask, edge_image_cropped)
    bestfit_circle = fit_circ_to_edge(membrane_edge, center_guess, 
                                      membrane_guess_r)
    
    # find interface relative to membrane edge 
    outer_mask = circular_mask(edge_image_cropped.shape, bestfit_circle[0:2],
                               bestfit_circle[2] - film_thick, False)
    interface_edge = np.logical_and(outer_mask, edge_image_cropped)

    # get interface height
    int_polar = interface_pol_coords(interface_edge, bestfit_circle[0:2])
    # select on phis within range
    int_polar = int_polar[:, np.logical_and(int_polar[1] > phi_range[0],
                                            int_polar[1] < phi_range[1])]
    avg_int_r = int_polar[0].mean()

    # new gradient fitting based measurement
    thick, ref_interface, ref_edge = thickness_by_fitting(crop(grad_image), 
                                                          bestfit_circle[0:2],
                                                          phi_range, r_range, 
                                                          avg_int_r)

    return membrane_edge, interface_edge, ref_edge, ref_interface, thick, \
        bestfit_circle



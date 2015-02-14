'''
Library for processing tangent angle data for line tension
measurements.

'''

import os
import numpy as np
import fourier_coefficients

def tangent_fluctuation_spectrum(q, gamma, kappa):
    '''
    Theoretical fluctuation spectrum

    gamma in units of kT/unit length (from q)
    kappa in units of kT unit length
    '''
    return 1./(gamma + kappa * q**2)


def load_tangents(tangent_dirs, lengths = False):
    '''
    Load data from a directory of tangent files.
    '''
    # expect string or list of strings
    if isinstance(tangent_dirs, basestring):
        tangent_dirs = [tangent_dirs]

    output = []
    for direct in tangent_dirs:
        if direct[-1] != '/': # check for trailing slash
            direct = direct + '/'
        for file in sorted(os.listdir(direct)):
            output.append(np.load(direct + file))

    if lengths:
        lengths = np.array([len(tgt) for tgt in output])
        print 'Min and max tangent lengths: ', lengths.min(), lengths.max()
        return output, lengths
    else:
        return np.array(output)


def analyze_cosines(tangents, length, n_max, px = 1, outfilebase = None):
    '''
    tangents: array of tangent angles, angle being tgt[:, -1]
    length: length of segment to fourier analyze (in pixels)
    n_max: maximum (integer) fourier mode
    px: conversion between pixels and length (eg. microns per pixel)
    outfilebase: file names for output

    Output: <a_q^2> in units of px
    '''
    length_dim = length * px
    n_images = len(tangents)
    
    cos_coeffs = np.zeros((n_images, n_max + 1))
    
    for tgt, ctr in zip(tangents, np.arange(n_images)):
        # pick central portion
        start_index = np.floor((len(tgt[:,-1]) - length) / 2.)
        tgt_cut = tgt[start_index : start_index + length, -1]
        a_ns, b_ns = fourier_coefficients.fourier_coeffs2(tgt_cut, 
                                                          n_max = n_max)
        cos_coeffs[ctr] = a_ns

    var_cos = np.var(cos_coeffs, axis = 0)
    var_cos_dim = var_cos * length_dim / 2.
    q_dim = np.arange(n_max + 1) * 2. * np.pi / length_dim

    output = np.vstack((q_dim, var_cos_dim))
    if outfilebase:
        np.save(outfilebase + '_var_cos.npy', output)
        np.save(outfilebase + '_cos_coeffs.npy', cos_coeffs)

    return output


def analyze_thickness_cosines(thicknesses, length, n_max, px = 1, 
                              outfilebase = None):
    '''
    tangents: array of tangent angles, angle being tgt[:, -1]
    length: length of segment to fourier analyze (in pixels)
    n_max: maximum (integer) fourier mode
    px: conversion between pixels and length (eg. microns per pixel)
    outfilebase: file names for output

    Output: <a_q^2> in units of px
    '''
    length_dim = length * px
    n_images = len(thicknesses)
    
    cos_coeffs = np.zeros((n_images, n_max + 1))
    
    for thick, ctr in zip(thicknesses, np.arange(n_images)):
        # pick central portion
        start_index = np.floor((len(thick) - length) / 2.)
        thick_cut = thick[start_index : start_index + length]
        a_ns, b_ns = fourier_coefficients.fourier_coeffs2(thick_cut, 
                                                          n_max = n_max)
        cos_coeffs[ctr] = a_ns

    var_cos = np.var(cos_coeffs, axis = 0)
    var_cos_dim = var_cos * length_dim / 2.
    q_dim = np.arange(n_max + 1) *  np.pi / length_dim

    output = np.vstack((q_dim, var_cos_dim))
    if outfilebase:
        np.save(outfilebase + '_var_cos.npy', output)
        np.save(outfilebase + '_cos_coeffs.npy', cos_coeffs)

    return output






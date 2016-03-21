#!/usr/bin/env python 

from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
import cPickle as pickle

import numpy as np
import os

from PIL import Image
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.config import config

from scipy.special import ndtri, ndtr
from scipy.stats import norm

### Load a model from disk
def load_file(filename):
    with open(filename, "rb") as f:
        p = pickle.load(f)
    if isinstance(p, Model):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return

    main_model = model.get_top_bricks()[0]
    # reset the random generator
    try:
        del main_model._theano_rng
        del main_model._theano_seed
    except AttributeError:
        # Do nothing
        pass
    main_model.seed_rng = np.random.RandomState(config.default_seed)
    return main_model

### 
def make_flat(z_dim, cols, rows, gaussian_prior=True, interleaves=0, shuffles=0):
    sqrt2 = 1.0
    def lerpTo(val, low, high):
        zeroToOne = np.clip((val + sqrt2) / (2 * sqrt2), 0, 1)
        return low + (high - low) * zeroToOne

    def lerp(val, low, high):
        return low + (high - low) * val

    def pol2cart(phi):
        x = np.cos(phi)
        y = np.sin(phi)
        return(x, y)

    #  http://stackoverflow.com/a/5347492
    # >>> interleave(np.array(range(6)))
    # array([0, 3, 1, 4, 2, 5])
    def interleave(offsets):
        shape = offsets.shape
        split_point = int(shape[0] / 2)
        a = np.array(offsets[:split_point])
        b = np.array(offsets[split_point:])
        c = np.empty(shape, dtype=a.dtype)
        c[0::2] = a
        c[1::2] = b
        return c

    def shuffle(offsets):
        np.random.shuffle(offsets)

    offsets = []
    for i in range(z_dim):
        offsets.append(pol2cart(i * np.pi / z_dim))
    offsets = np.array(offsets)

    for i in range(interleaves):
        offsets = interleave(offsets)

    for i in range(shuffles):
        shuffle(offsets)

    ul = []
    # range_high = 0.95
    # range_low = 1 - range_high
    range_high = 0.997  # 3 standard deviations
    range_low = 1 - range_high
    for r in range(rows):
        # xf = lerp(r / (rows-1.0), -1.0, 1.0)
        xf = (r - (rows / 2.0) + 0.5) / ((rows-1) / 2.0 + 0.5)
        for c in range(cols):
            # yf = lerp(c / (cols-1.0), -1.0, 1.0)
            yf = (c - (cols / 2.0) + 0.5) / ((cols-1) / 2.0 + 0.5)
            coords = map(lambda o: np.dot([xf, yf], o), offsets)
            ranged = map(lambda n:lerpTo(n, range_low, range_high), coords)
            # ranged = map(lambda n:lerpTo(n, range_low, range_high), [xf, yf])
            if(gaussian_prior):
                cdfed = map(ndtri, ranged)
            else:
                cdfed = ranged
            ul.append(cdfed)
    u = np.array(ul).reshape(rows,cols,z_dim).astype('float32')
    return u

def img_grid(arr, rows, cols, with_space):
    N = len(arr)
    channels, height, width = arr[0].shape

    total_height = rows * height
    total_width  = cols * width

    if with_space:
        total_height = total_height + (rows - 1)
        total_width  = total_width + (cols - 1)

    I = np.zeros((channels, total_height, total_width))
    I.fill(1)

    for i in xrange(rows*cols):
        if i < N:
            r = i // cols
            c = i % cols

            cur_im = arr[i]

            if cur_im is not None:
                if with_space:
                    offset_y, offset_x = r*height+r, c*width+c
                else:
                    offset_y, offset_x = r*height, c*width
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = cur_im
    
    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I)

    out = (255 * out).astype(np.uint8)
    return Image.fromarray(out)

def sample_at(model, locations):
    u_var = T.tensor3("u_var")
    sample = model.sample_given(u_var)
    do_sample = theano.function([u_var], outputs=sample, allow_input_downcast=True)
    #------------------------------------------------------------
    iters, dim = model.get_iters_and_dim()
    rows, cols, z_dim = locations.shape
    logging.info("Sampling {}x{} locations with {} iters and {}={} dim...".format(rows,cols,iters,dim,z_dim))
    numsamples = rows * cols
    u_list = np.zeros((iters, numsamples, dim))

    for y in range(rows):
        for x in range(cols):
            # xcur_ycur = np.random.normal(0, 1.0, (iters, 1, dim))
            xcur_ycur = np.zeros((iters, 1, dim))
            xcur_ycur[0,0,:] = locations[y][x].reshape(dim)
            n = y * cols + x
            # curu = rowmin
            u_list[:,n:n+1,:] = xcur_ycur

    samples = do_sample(u_list)
    print("Shape: {}".format(samples.shape))
    return samples

def sample_random_native(model, numsamples):
    n_samples = T.iscalar("n_samples")
    sample = model.sample(n_samples)
    do_sample = theano.function([n_samples], outputs=sample, allow_input_downcast=True)
    #------------------------------------------------------------
    logging.info("Sampling and saving images...")
    samples = do_sample(numsamples)
    return samples

def sample_random(model, numsamples):
    u_var = T.tensor3("u_var")
    sample = model.sample_given(u_var)
    do_sample = theano.function([u_var], outputs=sample, allow_input_downcast=True)
    #------------------------------------------------------------
    logging.info("Sampling images...")
    iters, dim = model.get_iters_and_dim()
    u = np.random.normal(0, 1, (iters, numsamples, dim))
    samples = do_sample(u)
    print("Shape: {}".format(samples.shape))
    return samples

def compute_splash(rows, cols, dim, space, anchors, spherical, gaussian):
    lerpv = get_lerpv_by_type(spherical, gaussian)

    u_list = np.zeros((rows, cols, dim))
    # compute anchors
    cur_anchor = 0
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space == 0:
                if anchors is not None and cur_anchor < len(anchors):
                    u_list[y,x,:] = anchors[cur_anchor]
                    cur_anchor = cur_anchor + 1
                else:
                    u_list[y,x,:] = np.random.normal(0,1, (1, dim))
    # interpolate horizontally
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space != 0:
                lastX = space * (x // space)
                nextX = lastX + space
                fracX = (x - lastX) / float(space)
#                 print("{} - {} - {}".format(lastX, nextX, fracX))
                u_list[y,x,:] = lerpv(fracX, u_list[y, lastX, :], u_list[y, nextX, :])
    # interpolate vertically
    for y in range(rows):
        for x in range(cols):
            if y%space != 0:
                lastY = space * (y // space)
                nextY = lastY + space
                fracY = (y - lastY) / float(space)
                u_list[y,x,:] = lerpv(fracY, u_list[lastY, x, :], u_list[nextY, x, :])

    u_grid = u_list.reshape(rows * cols, dim)

    return u_grid


def compute_splash_old2(rows, cols, dim, space):
    u_list = np.zeros((cols, rows, dim))

    # compute anchors
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space == 0:
                u_list[x,y,:] = np.random.uniform(0,1, (1, dim))
    # interpolate horizontally
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space != 0:
                lastX = space * (x // space)
                nextX = lastX + space
                fracX = (x - lastX) / float(space)
#                 print("{} - {} - {}".format(lastX, nextX, fracX))
                u_list[x,y,:] = lerp(fracX, u_list[lastX, y, :], u_list[nextX, y, :])
    # interpolate vertically
    for y in range(rows):
        for x in range(cols):
            if y%space != 0:
                lastY = space * (y // space)
                nextY = lastY + space
                fracY = (y - lastY) / float(space)
                u_list[x,y,:] = lerp(fracY, u_list[x, lastY, :], u_list[x, nextY, :])

    u_gau = ndtri(u_list).reshape(rows * cols, dim)

    return u_gau

def compute_splash_old(rows, cols, dim):
    u_list = np.zeros((cols, rows, dim))

    # compute anchors
    for y in range(rows):
        for x in range(cols):
            if y%2 == 0 and x%2 == 0:
                u_list[x,y,:] = np.random.uniform(0, 1, (1, dim))
    # interpolate horizontally
    for y in range(rows):
        for x in range(cols):
            if y%2 == 0 and x%2 == 1:
                u_list[x,y,:] = (u_list[x-1,y,:] + u_list[x+1,y,:]) / 2.0
    # interpolate vertically
    for y in range(rows):
        for x in range(cols):
            if y%2 == 1:
                u_list[x,y,:] = (u_list[x,y-1,:] + u_list[x,y+1,:]) / 2.0

    u_gau = ndtri(u_list).reshape(rows * cols, dim)

    return u_gau

def lerp(val, low, high):
    return low + (high - low) * val

# this is a placeholder for a future version of spherical interpolation. 
# I think the right thing to do would be to convert to n-sphere spherical
# coordinates and interpolate there.
# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
def lerp_circle(val, low, high):
    # first compute the interpolated length
    rad_low = np.linalg.norm(low)
    rad_high = np.linalg.norm(high)
    rad_cur = lerp(val, rad_low, rad_high)

    # then compute the linearly interpolated vector but at unit length
    lerp_vec = lerp(val, low, high)
    rad_lerp_vec = np.linalg.norm(lerp_vec)
    unit_vec = np.nan_to_num(lerp_vec / rad_lerp_vec)

    # now just return the product of the length and direction
    return rad_cur * unit_vec

def lerp_gaussian(val, low, high):
    low_gau = norm.cdf(low)
    high_gau = norm.cdf(high)
    lerped_gau = lerp(val, low_gau, high_gau)
    return norm.ppf(lerped_gau)

def lerp_circle_gaussian(val, low, high):
    offset = norm.cdf(np.zeros_like(low))  # offset is just [0.5, 0.5, ...]
    low_gau_shifted = norm.cdf(low) - offset
    high_gau_shifted = norm.cdf(high) - offset
    circle_lerped_gau = lerp_circle(val, low_gau_shifted, high_gau_shifted)
    return norm.ppf(circle_lerped_gau + offset)

# http://stackoverflow.com/a/2880012/1010653
def slerp(val, low, high):
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def slerp_gaussian(val, low, high):
    offset = norm.cdf(np.zeros_like(low))  # offset is just [0.5, 0.5, ...]
    low_gau_shifted = norm.cdf(low) - offset
    high_gau_shifted = norm.cdf(high) - offset
    circle_lerped_gau = slerp(val, low_gau_shifted, high_gau_shifted)
    epsilon = 0.001
    clipped_sum = np.clip(circle_lerped_gau + offset, epsilon, 1.0 - epsilon)
    result = norm.ppf(clipped_sum)
    return result

def get_lerpv_by_type(spherical, gaussian):
    if spherical and gaussian:
        return slerp_gaussian
    elif spherical:
        return slerp
    elif gaussian:
        return lerp_gaussian
    else:
        return lerp

def compute_gradient(rows, cols, dim, analogy, anchors, spherical, gaussian):
    lerpv = get_lerpv_by_type(spherical, gaussian)
    hyper = False

    numsamples = rows * cols
    u_list = np.zeros((numsamples, dim))
    if anchors is not None:
        # xmin_ymin, xmax_ymin, xmin_ymax = anchors[0:3]
        xmin_ymin, xmin_ymax, xmax_ymin = anchors[0:3]
    else:
        xmin_ymin = np.random.normal(0, 1, dim)
        xmax_ymin = np.random.normal(0, 1, dim)
        xmin_ymax = np.random.normal(0, 1, dim)
    if(analogy):
        xmax_ymax = xmin_ymax + (xmax_ymin - xmin_ymin)
        if hyper:
            tl = xmin_ymin
            tr = xmax_ymin
            bl = xmin_ymax
            xmax_ymax = bl + (tr - tl)
            xmin_ymax = bl - (tr - tl)
            xmax_ymin = tr + (tl - bl)
            xmin_ymin = xmin_ymax + (xmax_ymin - xmax_ymax)
    elif anchors is not None:
        xmax_ymax = anchors[3]
    else:
        xmax_ymax = np.random.normal(0, 1, dim)

    for y in range(rows):
        y_frac = y / (rows - 1)
        xmin_ycur = lerpv(y_frac, xmin_ymin, xmin_ymax)
        xmax_ycur = lerpv(y_frac, xmax_ymin, xmax_ymax)
        for x in range(cols):
            x_frac = x / (cols - 1)
            xcur_ycur = lerpv(x_frac, xmin_ycur, xmax_ycur)
            n = y * cols + x
            u_list[n:n+1,:] = xcur_ycur

    return u_list

def sample_gradient(model, rows, cols, analogy, anchors):
    u_var = T.tensor3("u_var")
    sample = model.sample_given(u_var)
    do_sample = theano.function([u_var], outputs=sample, allow_input_downcast=True)
    #------------------------------------------------------------
    logging.info("Sampling gradient...")
    iters, dim = model.get_iters_and_dim()

    numsamples = rows * cols
    u_list = np.zeros((iters, numsamples, dim))
    if anchors:
        xmin_ymin, xmax_ymin, xmin_ymax = anchors[0:3]
    else:
        xmin_ymin = np.random.normal(0, 1, (iters, 1, dim))
        xmax_ymin = np.random.normal(0, 1, (iters, 1, dim))
        xmin_ymax = np.random.normal(0, 1, (iters, 1, dim))
    if(analogy):
        xmax_ymax = xmin_ymax + (xmax_ymin - xmin_ymin)
    elif anchors:
        xmax_ymax = anchors[3]
    else:
        xmax_ymax = np.random.normal(0, 1, (iters, 1, dim))
    # A xmax_ymax = np.random.normal(0, 1, (iters, 1, dim))
    # B xmax_ymax = xmin_ymax + (xmax_ymin - xmin_ymin)
    # C xmax_ymax = xmin_ymin
    # C xmin_ymax = xmax_ymin

    for y in range(rows):
        # xcur_ymin = ((1.0 * y * xmin_ymin) + ((rows - y - 1.0) * xmax_ymin)) / (rows - 1.0)
        # xcur_ymax = ((1.0 * y * xmin_ymax) + ((rows - y - 1.0) * xmax_ymax)) / (rows - 1.0)
        xmin_ycur = (((rows - y - 1.0) * xmin_ymin) + (1.0 * y * xmin_ymax)) / (rows - 1.0)
        xmax_ycur = (((rows - y - 1.0) * xmax_ymin) + (1.0 * y * xmax_ymax)) / (rows - 1.0)
        for x in range(cols):
            # xcur_ycur = ((1.0 * x * xcur_ymin) + ((cols - x - 1.0) * xcur_ymax)) / (cols - 1.0)
            xcur_ycur = (((cols - x - 1.0) * xmin_ycur) + (1.0 * x * xmax_ycur)) / (cols - 1.0)
            n = y * cols + x
            # curu = rowmin
            u_list[:,n:n+1,:] = xcur_ycur

    samples = do_sample(u_list)
    print("Shape: {}".format(samples.shape))
    return samples

def sample_at_new(model, locations):
    n_iter, rows, cols, z_dim = locations.shape
    flat_locations = locations.reshape(n_iter, rows*cols, z_dim)

    n_samples = T.iscalar("n_samples")
    u_var = T.matrix("u_var")
    samples_at = model.sample_at_new(n_samples, u_var)
    do_sample_at = theano.function([n_samples, u_var], outputs=samples_at, allow_input_downcast=True)
    #------------------------------------------------------------
    logging.info("Sampling and saving images...")
    samples = do_sample_at(rows*cols, flat_locations)
    return samples

def build_reconstruct_function(model):
    x = T.matrix('features')
    reconstruct_function = theano.function([x], model.reconstruct(x))
    return reconstruct_function

def reconstruct_image(reconstruct_function, source_im):
    recon_im, kterms = reconstruct_function(source_im)
    return recon_im, kterms

def build_reconstruct_terms_function(model):
    x = T.matrix('features')
    reconstruct_function = theano.function([x], model.reconstruct_terms(x))
    return reconstruct_function

def reconstruct_terms(reconstruct_function, source_im):
    recon_im, z_terms = reconstruct_function(source_im)
    return recon_im, z_terms


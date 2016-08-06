#!/usr/bin/env python 

from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
import cPickle as pickle

import numpy as np
import os

from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.config import config

from plat.interpolate import get_interpfn

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

def cs(cols, x):
    # compute coords
    scaledX = x * (cols - 1)
    intX = int(scaledX)
    nextX = intX + 1
    spaceX = 1.0 / (cols - 1)
    fracX = scaledX - intX
    print(scaledX, intX, nextX, spaceX, fracX)

def compute_splash_latent(rows, cols, y, x, anchors, spherical=True, gaussian=False):
    lerpv = get_interpfn(spherical, gaussian)
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)

    _, dim = anchors.shape
    u_list = np.zeros((rows, cols, dim))
    # compute anchors
    cur_anchor = 0
    for ys in range(rows):
        for xs in range(cols):
            if anchors is not None and cur_anchor < len(anchors):
                u_list[ys,xs,:] = anchors[cur_anchor]
                cur_anchor = cur_anchor + 1
            else:
                u_list[ys,xs,:] = np.random.normal(0,1, (1, dim))

    # compute coords
    spaceX = 1.0 / (cols - 1)
    scaledX = x * (cols - 1)
    intX = int(scaledX)
    nextX = intX + 1
    fracX = scaledX - intX

    spaceY = 1.0 / (rows - 1)
    scaledY = y * (rows - 1)
    intY = int(scaledY)
    nextY = intY + 1
    fracY = scaledY - intY

    # hack to allow x=1.0 and y=1.0
    if fracX == 0:
        nextX = intX
    if fracY == 0:
        nextY = intY

    h1 = lerpv(fracX, u_list[intY, intX, :], u_list[intY, nextX, :])
    h2 = lerpv(fracX, u_list[nextY, intX, :], u_list[nextY, nextX, :])

    # interpolate vertically
    result = lerpv(fracY, h1, h2)
    if np.isnan(result[0]):
        print("NAN FOUND")
        print("h1: ", h1)
        print("h2: ", h2)
        print("fracx,y", fracX, fracY)
        print("xvars", spaceX, scaledX, intX, nextX, fracX)
        print("yvars", spaceY, scaledY, intY, nextY, fracY)
        print("inputs", x, y, rows, cols)
    return result

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


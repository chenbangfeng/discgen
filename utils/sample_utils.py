#!/usr/bin/env python

"""Plots model samples."""
import argparse

import theano
from blocks.bricks import Random
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.select import Selector
from blocks.serialization import load
from blocks.utils import shared_floatx
from blocks.config import config
from theano import tensor
from utils.modelutil import make_flat, compute_gradient, compute_splash, img_grid
import numpy as np
import random
import sys
import json
from scipy.misc import imread, imsave

from discgen.utils import plot_image_grid

from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from discgen.utils import Colorize

def anchors_from_image(fname, channels=3, image_size=(64,64)):
    rawim = imread(fname);
    if(channels == 1):
        im_height, im_width, im_channels = rawim.shape
        mixedim = rawim
    else:
        im_height, im_width, im_channels = rawim.shape
        mixedim = np.asarray([rawim[:,:,0], rawim[:,:,1], rawim[:,:,2]])

    pairs = []
    target_shape = (channels, image_size[0], image_size[1])
    height, width = image_size

    # first build a list of num images in datastream
    datastream_images = []
    steps_y = int(im_height / height)
    steps_x = int(im_width / width)
    # while cur_x + width <= im_width and len(datastream_images) < num:
    for j in range(steps_y):
        cur_y = j * height
        for i in range(steps_x):
            cur_x = i * width
            if(channels == 1):
                entry = (mixedim[cur_y:cur_y+height, cur_x:cur_x+width] / 255.0).astype('float32')
            else:
                entry = (mixedim[0:im_channels, cur_y:cur_y+height, cur_x:cur_x+width] / 255.0).astype('float32')
            datastream_images.append(entry)

    return steps_y, steps_x, datastream_images

def get_image_encoder_function(model):
    selector = Selector(model.top_bricks)
    encoder_convnet, = selector.select('/encoder_convnet').bricks
    encoder_mlp, = selector.select('/encoder_mlp').bricks

    print('Building computation graph...')
    x = tensor.tensor4('features')
    phi = encoder_mlp.apply(encoder_convnet.apply(x).flatten(ndim=2))
    nlat = encoder_mlp.output_dim // 2
    mu_phi = phi[:, :nlat]
    log_sigma_phi = phi[:, nlat:]
    epsilon = Random().theano_rng.normal(size=mu_phi.shape, dtype=mu_phi.dtype)
    z = mu_phi + epsilon * tensor.exp(log_sigma_phi)
    computation_graph = ComputationGraph([x, z])

    print('Compiling reconstruction function...')
    encoder_function = theano.function(
        computation_graph.inputs, computation_graph.outputs)
    return encoder_function

def get_image_vectors(model, images):
    encoder_function = get_image_encoder_function(model)
    print('Encoding...')
    examples, latents = encoder_function(images)
    return latents

def get_json_vectors(filename):
    with open(filename) as json_file:
        json_data = json.load(json_file)
    return np.array(json_data)

def offset_from_string(x_indices_str, offsets, dim):
    x_offset = np.zeros((dim,))
    if x_indices_str[0] == ",":
        x_indices_str = x_indices_str[1:]
    x_indices = map(int, x_indices_str.split(","))
    for x_index in x_indices:
        if x_index < 0:
            scaling = -1.0
            x_index = -x_index
        else:
            scaling = 1.0
        x_offset += scaling * offsets[x_index]
    return x_offset


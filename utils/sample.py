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

from discgen.utils import plot_image_grid

from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from discgen.utils import Colorize

def get_anchor_images(dataset, split, numanchors, allowed, prohibited, color_convert=False, include_targets=True):
    sources = ('features', 'targets') if include_targets else ('features',)
    if split == "all":
        splits = ('train', 'valid', 'test')
    elif split == "nontrain":
        splits = ('valid', 'test')
    else:
        splits = (split,)

    dataset_fname = find_in_data_path("{}.hdf5".format(dataset))
    datastream = H5PYDataset(dataset_fname, which_sets=splits,
                             sources=sources)
    datastream.default_transformers = uint8_pixels_to_floatX(('features',))

    train_stream = DataStream.default_stream(
        dataset=datastream,
        iteration_scheme=SequentialExampleScheme(datastream.num_examples))

    it = train_stream.get_epoch_iterator()    

    anchors = []
    while len(anchors) < numanchors:
        cur = it.next()
        candidate_passes = True
        if allowed:
            for p in allowed:
                if(cur[1][p] != 1):
                    candidate_passes = False
        if prohibited:
            for p in prohibited:
                if(cur[1][p] != 0):
                    candidate_passes = False

        if candidate_passes:
            if color_convert:
                anchors.append(np.tile(cur[0].reshape(1, 64, 64), (3, 1, 1)))
            else:
                anchors.append(cur[0])

    return np.array(anchors)

def get_image_vectors(model, images):
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

    print('Encoding...')
    examples, latents = encoder_function(images)
    return latents

# returns new version of images, rows, cols
def add_shoulders(images, anchor_images, rows, cols):
    ncols = cols + 2
    nimages = []
    cur_im = 0
    for j in range(rows):
        for i in range(ncols):
            if i == 0 and j == 0:
                nimages.append(anchor_images[0])
            elif i == 0 and j == rows-1:
                nimages.append(anchor_images[1])
            elif i == ncols-1 and j == 0:
                nimages.append(anchor_images[2])
            elif i > 0 and i < ncols-1:
                nimages.append(images[cur_im])
                cur_im = cur_im + 1
            else:
                nimages.append(None)
    return nimages, rows, ncols

# returns list of latent variables to support rows x cols 
def generate_latent_grid(z_dim, rows, cols, flat, gradient, spherical, gaussian, anchors, anchor_images, splash, spacing, analogy):
    if flat:
        z = make_flat(z_dim, cols, rows)
    elif gradient:
        z = compute_gradient(rows, cols, z_dim, analogy, anchors, spherical, gaussian)
    elif splash:
        z = compute_splash(rows, cols, z_dim, spacing, anchors, spherical, gaussian)
    else:
        # TODO: non-gaussian version
        z = np.random.normal(loc=0, scale=1, size=(rows * cols, z_dim))

    return z

def grid_from_latents(z, model, rows, cols, anchor_images, tight, shoulders, save_path):
    selector = Selector(model.top_bricks)
    decoder_mlp, = selector.select('/decoder_mlp').bricks
    decoder_convnet, = selector.select('/decoder_convnet').bricks

    print('Building computation graph...')
    sz = shared_floatx(z)
    mu_theta = decoder_convnet.apply(
        decoder_mlp.apply(sz).reshape(
            (-1,) + decoder_convnet.get_dim('input_')))
    computation_graph = ComputationGraph([mu_theta])

    print('Compiling sampling function...')
    sampling_function = theano.function(
        computation_graph.inputs, computation_graph.outputs[0])

    print('Sampling...')
    samples = sampling_function()

    if shoulders:
        samples, rows, cols = add_shoulders(samples, anchor_images, rows, cols)

    print('Preparing image grid...')
    img = img_grid(samples, rows, cols, not tight)
    img.save(save_path)


def reconstruct_grid(model, rows, cols, flat, gradient, spherical, gaussian, anchors, anchor_images, splash, spacing, analogy, tight, shoulders, save_path):
    selector = Selector(model.top_bricks)
    decoder_mlp, = selector.select('/decoder_mlp').bricks
    z_dim = decoder_mlp.input_dim
    z = generate_latent_grid(z_dim, rows, cols, flat, gradient, spherical, gaussian, anchors, anchor_images, splash, spacing, analogy)
    grid_from_latents(z, model, rows, cols, anchor_images, tight, shoulders, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--rows", type=int, default=5,
                        help="number of rows of samples to display")
    parser.add_argument("--cols", type=int, default=5,
                        help="number of columns of samples to display")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the generated samples")
    parser.add_argument('--flat', dest='flat', default=False, action='store_true')
    parser.add_argument('--analogy', dest='analogy', default=False, action='store_true')
    parser.add_argument('--gradient', dest='gradient', default=False, action='store_true')
    parser.add_argument('--linear', dest='linear', default=False, action='store_true')
    parser.add_argument('--gaussian', dest='gaussian', default=False, action='store_true')
    parser.add_argument('--tight', dest='tight', default=False, action='store_true')
    parser.add_argument("--seed", type=int,
                default=None, help="Optional random seed")
    parser.add_argument('--splash', dest='splash', default=False, action='store_true')
    parser.add_argument("--spacing", type=int, default=3,
                        help="spacing of splash grid, w & h must be multiples +1")
    parser.add_argument('--anchors', dest='anchors', default=False, action='store_true',
                        help="use reconstructed images instead of random ones")
    parser.add_argument("--numanchors", type=int, default=150,
                        help="number of anchors to generate")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Dataset for anchors.")
    parser.add_argument('--color-convert', dest='color_convert',
                        default=False, action='store_true',
                        help="Convert source dataset to color from grayscale.")
    parser.add_argument('--split', dest='split', default="all",
                        help="Which split to use from the dataset (train/nontrain/valid/test/any).")
    parser.add_argument("--allowed", dest='allowed', type=str, default=None,
                        help="Only allow whitelisted labels L1,L2,...")
    parser.add_argument("--prohibited", dest='prohibited', type=str, default=None,
                        help="Only allow blacklisted labels L1,L2,...")
    parser.add_argument('--passthrough', dest='passthrough', default=False, action='store_true',
                        help="Use originals instead of reconstructions")
    parser.add_argument('--shoulders', dest='shoulders', default=False, action='store_true',
                        help="Append anchors to left/right columns")
    parser.add_argument('--encoder', dest='encoder', default=False, action='store_true',
                        help="Ouput dataset as encoded vectors")
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)

    anchor_images = None
    if args.anchors:
        allowed = None
        prohibited = None
        if(args.allowed):
            allowed = map(int, args.allowed.split(","))
        if(args.prohibited):
            prohibited = map(int, args.prohibited.split(","))
        anchor_images = get_anchor_images(args.dataset, args.split, args.numanchors, allowed, prohibited, args.color_convert)

    if args.passthrough:
        print('Preparing image grid...')
        img = img_grid(anchor_images, args.rows, args.cols, not args.tight)
        img.save(args.save_path)
        sys.exit(0)

    print('Loading saved model...')
    model = Model(load(args.model).algorithm.cost)

    if anchor_images is not None:
        anchors = get_image_vectors(model, anchor_images)
    else:
        anchors = None

    selector = Selector(model.top_bricks)
    decoder_mlp, = selector.select('/decoder_mlp').bricks
    z_dim = decoder_mlp.input_dim
    z = generate_latent_grid(z_dim, args.rows, args.cols, args.flat, args.gradient, not args.linear, args.gaussian,
            anchors, anchor_images, args.splash, args.spacing, args.analogy)
    grid_from_latents(z, model, args.rows, args.cols, anchor_images, args.tight, args.shoulders, args.save_path)
    # reconstruct_grid(model, args.rows, args.cols, args.flat, args.gradient, not args.linear, args.gaussian,
    #     anchors, anchor_images, args.splash, args.spacing, args.analogy, args.tight, args.shoulders, args.save_path)

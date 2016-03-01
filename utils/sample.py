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
from theano import tensor
from utils.modelutil import make_flat, compute_gradient, compute_splash, img_grid, get_anchor_images
import numpy as np
import random
import sys

from discgen.utils import plot_image_grid

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

def reconstruct_grid(model, rows, cols, flat, gradient, spherical, gaussian, anchors, splash, spacing, analogy, tight, save_path):
    selector = Selector(model.top_bricks)
    decoder_mlp, = selector.select('/decoder_mlp').bricks
    decoder_convnet, = selector.select('/decoder_convnet').bricks

    print('Building computation graph...')

    z_dim = decoder_mlp.input_dim
    if flat:
        z = shared_floatx(make_flat(z_dim, cols, rows))
    elif gradient:
        z = shared_floatx(compute_gradient(rows, cols, z_dim, analogy, anchors, spherical, gaussian))
    elif splash:
        z = shared_floatx(compute_splash(rows, cols, z_dim, spacing, anchors, spherical, gaussian))
    else:
        z = Random().theano_rng.normal(size=(rows * cols, z_dim),
                                       dtype=theano.config.floatX)
    mu_theta = decoder_convnet.apply(
        decoder_mlp.apply(z).reshape(
            (-1,) + decoder_convnet.get_dim('input_')))
    computation_graph = ComputationGraph([mu_theta])

    print('Compiling sampling function...')
    sampling_function = theano.function(
        computation_graph.inputs, computation_graph.outputs[0])

    print('Sampling...')
    samples = sampling_function()

    print('Preparing image grid...')
    img = img_grid(samples, rows, cols, not tight)
    img.save(save_path)
    # img.save("{0}/{1}.png".format(subdir, filename))
    # plot_image_grid(samples, nrows, ncols, save_path)


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
    parser.add_argument('--spherical', dest='spherical', default=False, action='store_true')
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
    parser.add_argument('--split', dest='split', default="all",
                        help="Which split to use from the dataset (train/valid/test/any).")
    parser.add_argument("--allowed", dest='allowed', type=str, default=None,
                        help="Only allow whitelisted labels L1,L2,...")
    parser.add_argument("--prohibited", dest='prohibited', type=str, default=None,
                        help="Only allow blacklisted labels L1,L2,...")
    parser.add_argument('--passthrough', dest='passthrough', default=False, action='store_true',
                        help="Use originals instead of reconstructions")
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
        anchor_images = get_anchor_images(args.dataset, args.split, args.numanchors, allowed, prohibited)

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

    reconstruct_grid(model, args.rows, args.cols, args.flat, args.gradient, args.spherical, args.gaussian,
        anchors, args.splash, args.spacing, args.analogy, args.tight, args.save_path)

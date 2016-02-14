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
from utils.modelutil import make_flat, compute_gradient, compute_splash, img_grid
import numpy as np
import random

from discgen.utils import plot_image_grid

def main(saved_model_path, rows, cols, flat, gradient, splash, analogy, tight, save_path):
    print('Loading saved model...')
    model = Model(load(saved_model_path).algorithm.cost)
    selector = Selector(model.top_bricks)
    decoder_mlp, = selector.select('/decoder_mlp').bricks
    decoder_convnet, = selector.select('/decoder_convnet').bricks

    print('Building computation graph...')

    z_dim = decoder_mlp.input_dim
    if flat:
        z = shared_floatx(make_flat(z_dim, cols, rows))
    elif gradient:
        z = shared_floatx(compute_gradient(rows, cols, z_dim, analogy, None))
    elif splash:
        z = shared_floatx(compute_splash(rows, cols, z_dim, 3))
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
    parser.add_argument("saved_model_path", type=str,
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
    parser.add_argument('--tight', dest='tight', default=False, action='store_true')
    parser.add_argument("--seed", type=int,
                default=None, help="Optional random seed")
    parser.add_argument('--splash', dest='splash', default=False, action='store_true')
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)

    main(args.saved_model_path, args.rows, args.cols, args.flat, args.gradient, args.splash, args.analogy, args.tight, args.save_path)

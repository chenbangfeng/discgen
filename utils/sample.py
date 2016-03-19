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
from scipy.misc import imread, imsave

from discgen.utils import plot_image_grid

from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from discgen.utils import Colorize

def get_anchor_images(dataset, split, offset, numanchors, allowed, prohibited, color_convert=False, include_targets=True):
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
    for i in range(offset):
        cur = it.next()
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

def surround_anchors(rows, cols, anchors, rand_anchors):
    newanchors = []
    cur_anc = 0
    cur_rand = 0
    for r in range(rows):
        for c in range(cols):
            if r == 0 or c == 0 or r == rows-1 or c == cols-1:
                newanchors.append(rand_anchors[cur_rand])
                cur_rand = cur_rand + 1
            else:
                newanchors.append(anchors[cur_anc])
                cur_anc = cur_anc + 1
    return newanchors

def anchors_from_image(fname, channels=3, image_size=(64,64)):
    rawim = imread(fname);
    if(channels == 1):
        im_height, im_width = rawim.shape
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
    parser.add_argument('--encircle', dest='encircle', default=False, action='store_true')
    parser.add_argument('--partway', dest='partway', type=float, default=None)
    parser.add_argument("--spacing", type=int, default=3,
                        help="spacing of splash grid, w & h must be multiples +1")
    parser.add_argument('--anchors', dest='anchors', default=False, action='store_true',
                        help="use reconstructed images instead of random ones")
    parser.add_argument('--anchor-image', dest='anchor_image', default=None,
                        help="use image as source of anchors")
    parser.add_argument("--numanchors", type=int, default=150,
                        help="number of anchors to generate")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Dataset for anchors.")
    parser.add_argument('--color-convert', dest='color_convert',
                        default=False, action='store_true',
                        help="Convert source dataset to color from grayscale.")
    parser.add_argument('--split', dest='split', default="all",
                        help="Which split to use from the dataset (train/nontrain/valid/test/any).")
    parser.add_argument("--offset", type=int, default=0,
                        help="data offset to skip")
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
        anchor_images = get_anchor_images(args.dataset, args.split, args.offset, args.numanchors, allowed, prohibited, args.color_convert)

    if args.anchor_image is not None:
        _, _, anchor_images = anchors_from_image(args.anchor_image)

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
    if (args.partway is not None) or args.encircle or (args.splash and anchors is None):
        srows=((args.rows // args.spacing) + 1)
        scols=((args.cols // args.spacing) + 1)
        rand_anchors = generate_latent_grid(z_dim, rows=srows, cols=scols, flat=False, gradient=False,
            spherical=False, gaussian=False, anchors=None, anchor_images=None, splash=False, spacing=args.spacing, analogy=False)
        if args.partway is not None:
            l = len(rand_anchors)
            clipped_anchors = anchors[:l]
            anchors = (1.0 - args.partway) * rand_anchors + args.partway * clipped_anchors
        elif args.encircle:
            anchors = surround_anchors(srows, scols, anchors, rand_anchors)
        else:
            anchors = rand_anchors
    z = generate_latent_grid(z_dim, args.rows, args.cols, args.flat, args.gradient, not args.linear, args.gaussian,
            anchors, anchor_images, args.splash, args.spacing, args.analogy)
    grid_from_latents(z, model, args.rows, args.cols, anchor_images, args.tight, args.shoulders, args.save_path)
    # reconstruct_grid(model, args.rows, args.cols, args.flat, args.gradient, not args.linear, args.gaussian,
    #     anchors, anchor_images, args.splash, args.spacing, args.analogy, args.tight, args.shoulders, args.save_path)

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
from sample_utils import anchors_from_image, get_image_encoder_function, get_image_vectors, get_json_vectors, offset_from_string

from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from discgen.utils import Colorize

g_image_size = 128

def get_dataset_iterator(dataset, split, include_targets=False):
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
    return it

def get_anchor_images(dataset, split, offset, stepsize, numanchors, allowed, prohibited, image_size, color_convert=False, include_targets=True):
    it = get_dataset_iterator(dataset, split, include_targets)

    anchors = []
    for i in range(offset):
        cur = it.next()
    while len(anchors) < numanchors:
        cur = it.next()
        for s in range(stepsize-1):
            it.next()
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
                anchors.append(np.tile(cur[0].reshape(1, image_size, image_size), (3, 1, 1)))
            else:
                anchors.append(cur[0])

    return np.array(anchors)

# returns new version of images, rows, cols
def add_shoulders(images, anchor_images, rows, cols):
    ncols = cols + 2
    nimages = []
    cur_im = 0
    n_anchors = len(anchor_images)
    for j in range(rows):
        for i in range(ncols):
            if i == 0 and j == 0 and n_anchors > 0:
                nimages.append(anchor_images[0])
            elif i == 0 and j == rows-1 and n_anchors > 1:
                nimages.append(anchor_images[1])
            elif i == ncols-1 and j == 0 and n_anchors > 2:
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

def samples_from_latents(z, model):
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
    return samples

def grid_from_latents(z, model, rows, cols, anchor_images, tight, shoulders, save_path):
    samples = samples_from_latents(z, model)

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

def vector_to_json_array(v):
    return json.dumps(v.tolist())

def output_vectors(vectors):
    print("VECTOR OUTPUT BEGIN")
    print("JSON#[")
    for v in vectors[:-1]:
        print("JSON#{},".format(vector_to_json_array(v)))
    for v in vectors[-1:]:
        print("JSON#{}".format(vector_to_json_array(v)))
    print("JSON#]")
    print("VECTOR OUTPUT END")

def anchors_from_offsets(anchor, offsets, x_indices_str, y_indices_str, x_minscale, y_minscale, x_maxscale, y_maxscale):
    dim = len(anchor)
    x_offset = offset_from_string(x_indices_str, offsets, dim)
    y_offset = offset_from_string(y_indices_str, offsets, dim)

    newanchors = []
    newanchors.append(anchor + x_minscale * x_offset + y_minscale * y_offset)
    newanchors.append(anchor + x_minscale * x_offset + y_maxscale * y_offset)
    newanchors.append(anchor + x_maxscale * x_offset + y_minscale * y_offset)
    newanchors.append(anchor + x_maxscale * x_offset + y_maxscale * y_offset)
    return np.array(newanchors)

def get_global_offset(offsets, indices_str, scale):
    dim = len(offsets[0])
    global_offset = offset_from_string(indices_str, offsets, dim)
    return scale * global_offset

def stream_output_vectors(model, dataset, split, color_convert=False):
    encoder_function = get_image_encoder_function(model)

    it = get_dataset_iterator(dataset, split)
    batch_size = 20
    done = False

    print("VECTOR OUTPUT BEGIN")
    print("JSON#[")

    while not done:
        anchors = []
        try:
            for i in range(batch_size):
                cur = it.next()
                if color_convert:
                    anchors.append(np.tile(cur[0].reshape(1, g_image_size, g_image_size), (3, 1, 1)))
                else:
                    anchors.append(cur[0])
            anchors_input = np.array(anchors)
            examples, latents = encoder_function(anchors_input)
            for v in latents:
                print("JSON#{},".format(vector_to_json_array(v)))
        except StopIteration:
            # process any leftovers
            if len(anchors) > 0:
                anchors_input = np.array(anchors)
                examples, latents = encoder_function(anchors_input)
                # end cut-n-paste
                for v in latents[:-1]:
                    print("JSON#{},".format(vector_to_json_array(v)))
                for v in latents[-1:]:
                    print("JSON#{}".format(vector_to_json_array(v)))
            done = True

    # for v in vectors[-1:]:
    #     print("{}".format(vector_to_json_array(v)))

    print("JSON#]")
    print("VECTOR OUTPUT END")

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
    parser.add_argument('--global-offset', dest='global_offset', default=None,
                        help="use json file as source of global offsets")
    parser.add_argument('--global-indices', dest='global_indices', default=None, type=str,
                        help="offset indices to apply globally")
    parser.add_argument('--global-scale', dest='global_scale', default=1.0, type=float,
                        help="scaling factor for global offset")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument('--anchor-offset-x', dest='anchor_offset_x', default="5", type=str,
                        help="which indices to combine for x offset")
    parser.add_argument('--anchor-offset-y', dest='anchor_offset_y', default="39", type=str,
                        help="which indices to combine for y offset")
    parser.add_argument('--anchor-offset-x-minscale', dest='anchor_offset_x_minscale', default=0, type=float,
                        help="scaling factor for min x offset")
    parser.add_argument('--anchor-offset-y-minscale', dest='anchor_offset_y_minscale', default=0, type=float,
                        help="scaling factor for min y offset")
    parser.add_argument('--anchor-offset-x-maxscale', dest='anchor_offset_x_maxscale', default=2.0, type=float,
                        help="scaling factor for min x offset")
    parser.add_argument('--anchor-offset-y-maxscale', dest='anchor_offset_y_maxscale', default=2.0, type=float,
                        help="scaling factor for min y offset")
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
    parser.add_argument('--anchor-vectors', dest='anchor_vectors', default=None,
                        help="use json file as source of anchors")
    parser.add_argument('--invert-anchors', dest='invert_anchors',
                        default=False, action='store_true',
                        help="Use antipode of given anchors.")
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
    parser.add_argument("--stepsize", type=int, default=1,
                        help="data step size from offset")
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
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
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
        anchor_images = get_anchor_images(args.dataset, args.split, args.offset, args.stepsize, args.numanchors, allowed, prohibited, args.image_size, args.color_convert)

    if args.anchor_image is not None:
        _, _, anchor_images = anchors_from_image(args.anchor_image, image_size=(args.image_size, args.image_size))
        if args.offset > 0:
            anchor_images = anchor_images[args.offset:]

    if args.passthrough:
        print('Preparing image grid...')
        img = img_grid(anchor_images, args.rows, args.cols, not args.tight)
        img.save(args.save_path)
        sys.exit(0)

    print('Loading saved model...')
    model = Model(load(args.model).algorithm.cost)

    if anchor_images is not None:
        anchors = get_image_vectors(model, anchor_images)
    elif args.anchor_vectors is not None:
        anchors = get_json_vectors(args.anchor_vectors)
    else:
        anchors = None

    if args.invert_anchors:
        anchors = -1 * anchors

    if args.encoder:
        if anchors is not None:
            output_vectors(anchors)
        else:
            stream_output_vectors(model, args.dataset, args.split)
        sys.exit(0)

    global_offset = None
    if args.anchor_offset is not None:
        # compute anchors as offsets from existing anchor
        offsets = get_json_vectors(args.anchor_offset)
        anchors = anchors_from_offsets(anchors[0], offsets, args.anchor_offset_x, args.anchor_offset_y,
            args.anchor_offset_x_minscale, args.anchor_offset_y_minscale, args.anchor_offset_x_maxscale, args.anchor_offset_y_maxscale)

    if args.global_offset is not None:
        offsets = get_json_vectors(args.global_offset)
        global_offset =  get_global_offset(offsets, args.global_indices, args.global_scale)

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
    if global_offset is not None:
        z = z + global_offset

    grid_from_latents(z, model, args.rows, args.cols, anchor_images, args.tight, args.shoulders, args.save_path)
    # reconstruct_grid(model, args.rows, args.cols, args.flat, args.gradient, not args.linear, args.gaussian,
    #     anchors, anchor_images, args.splash, args.spacing, args.analogy, args.tight, args.shoulders, args.save_path)

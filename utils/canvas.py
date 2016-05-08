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
from sample_utils import anchors_from_image, get_image_vectors, get_json_vectors, offset_from_string

from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from discgen.utils import Colorize

from PIL import Image

channels = 4

# modified from http://stackoverflow.com/a/3375291/1010653
def alpha_composite(src, src_mask, dst):
    '''
    Return the alpha composite of src and dst.

    Parameters:
    src -- RGBA in range 0.0 - 1.0
    dst -- RGBA in range 0.0 - 1.0

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    out = np.empty(dst.shape, dtype = 'float')
    alpha = np.index_exp[3:, :, :]
    rgb = np.index_exp[:3, :, :]
    epsilon = 0.001
    src_a = np.maximum(src_mask, epsilon)
    dst_a = np.maximum(dst[alpha], epsilon)
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = np.seterr(invalid = 'ignore')
    out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
    np.seterr(**old_setting)
    np.clip(out,0,1.0)
    return out

gsize = 128
gsize2 = gsize/2

class Canvas:
    """Simple Canvas Thingy"""

    def __init__(self, width, height, xmin, xmax, ymin, ymax):
        self.pixels = np.zeros((channels, height, width))
        self.canvas_xmin = 0
        self.canvas_xmax = width
        self.canvas_ymin = 0
        self.canvas_ymax = height
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.canvas_xspread = self.canvas_xmax - self.canvas_xmin
        self.canvas_yspread = self.canvas_ymax - self.canvas_ymin
        self.xspread = self.xmax - self.xmin
        self.yspread = self.ymax - self.ymin
        self.xspread_ratio = float(self.canvas_xspread) / self.xspread
        self.yspread_ratio = float(self.canvas_yspread) / self.yspread

        _, _, mask_images = anchors_from_image("mask/rounded_mask{}.png".format(gsize), image_size=(gsize, gsize))
        self.mask = mask_images[0][0]

    # To map
    # [A, B] --> [a, b]
    # use this formula
    # (val - A)*(b-a)/(B-A) + a
    # A,B is virtual
    # a,b is canvas
    def map_to_canvas(self, x, y):
        new_x = int((x - self.xmin) * self.xspread_ratio + self.canvas_xmin)
        new_y = int((y - self.ymin) * self.yspread_ratio + self.canvas_ymin)
        return new_x, new_y

    def place_square(self, x, y):
        square = np.zeros((channels, gsize, gsize))
        square.fill(1)
        cx, cy = self.map_to_canvas(x, y)
        self.pixels[:, (cy-gsize2):(cy+gsize2), (cx-gsize2):(cx+gsize2)] = square

    def check_bounds(self, cx, cy):
        border = gsize2
        if (cx < self.canvas_xmin + border) or (cy < self.canvas_ymin + border) or (cx >= self.canvas_xmax - border) or (cy >= self.canvas_ymax - border):
            return False
        return True

    def place_image(self, im, x, y):
        square = im
        cx, cy = self.map_to_canvas(x, y)
        if self.check_bounds(cx, cy):
            self.pixels[:, (cy-gsize2):(cy+gsize2), (cx-gsize2):(cx+gsize2)] = \
                alpha_composite(im, self.mask, self.pixels[:, (cy-gsize2):(cy+gsize2), (cx-gsize2):(cx+gsize2)])

    def save(self, save_path):
        out = np.dstack(self.pixels)
        out = (255 * out).astype(np.uint8)
        img = Image.fromarray(out)
        img.save(save_path)

def images_from_latents(z, model):
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

def apply_anchor_offsets(anchor, offsets, a, b, a_indices_str, b_indices_str):
    dim = len(anchor)
    a_offset = offset_from_string(a_indices_str, offsets, dim)
    b_offset = offset_from_string(b_indices_str, offsets, dim)
    new_anchor = anchor + a * a_offset + b * b_offset
    # print(a, a*a_offset)
    return new_anchor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--width", type=int, default=512,
                        help="width of canvas to render in pixels")
    parser.add_argument("--height", type=int, default=512,
                        help="height of canvas to render in pixels")
    parser.add_argument("--xmin", type=int, default=0,
                        help="min x in virtual space")
    parser.add_argument("--xmax", type=int, default=100,
                        help="max x in virtual space")
    parser.add_argument("--ymin", type=int, default=0,
                        help="min y in virtual space")
    parser.add_argument("--ymax", type=int, default=100,
                        help="max y in virtual space")
    parser.add_argument("--save-path", type=str, default="out.png",
                        help="where to save the generated samples")
    parser.add_argument("--seed", type=int,
                default=None, help="Optional random seed")
    parser.add_argument('--anchor-image', dest='anchor_image', default=None,
                        help="use image as source of anchors")
    parser.add_argument('--layout', dest='layout', default=None,
                        help="layout json file")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=100,
                        help="number of images to decode at once")
    parser.add_argument('--passthrough', dest='passthrough', default=False, action='store_true',
                        help="Use originals instead of reconstructions")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument('--anchor-offset-a', dest='anchor_offset_a', default="42", type=str,
                        help="which indices to combine for offset a")
    parser.add_argument('--anchor-offset-b', dest='anchor_offset_b', default="31", type=str,
                        help="which indices to combine for offset b")
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)

    anchor_images = None
    if args.anchor_image is not None:
        _, _, anchor_images = anchors_from_image(args.anchor_image, image_size=(gsize, gsize))

    if not args.passthrough:
        print('Loading saved model...')
        model = Model(load(args.model).algorithm.cost)

        if anchor_images is not None:
            # anchors = anchor_images
            anchors = get_image_vectors(model, anchor_images)

    anchor_offsets = None
    if args.anchor_offset is not None:
        # compute anchors as offsets from existing anchor
        anchor_offsets = get_json_vectors(args.anchor_offset)

    encodes = {}

    canvas = Canvas(args.width, args.height, args.xmin, args.xmax, args.ymin, args.ymax)
    workq = []

    if args.layout:
        with open(args.layout) as json_file:
            layout_data = json.load(json_file)
        xy = np.array(layout_data["xy"])
        roots = layout_data["r"]
        for i, pair in enumerate(xy):
            x = pair[0] * canvas.xmax
            y = pair[1] * canvas.ymax
            a = pair[0]
            b = pair[1]
            r = roots[i]
            if args.passthrough:
                output_image = anchor_images[r]
                canvas.place_image(output_image, x, y)
            else:
                if anchor_offsets is not None:
                    z = apply_anchor_offsets(anchors[r], anchor_offsets, a, b, args.anchor_offset_a, args.anchor_offset_b)
                else:
                    z = anchors[r]
                workq.append({
                        "z": z,
                        "x": x,
                        "y": y
                    })

    while(len(workq) > 0):
        curq = workq[:args.batch_size]
        workq = workq[args.batch_size:]
        latents = [e["z"] for e in curq]
        images = images_from_latents(latents, model)
        # images = latents
        for i in range(len(curq)):
            canvas.place_image(images[i], curq[i]["x"], curq[i]["y"])

    # canvas.place_image(anchor_images[1], 50, 50)
    # canvas.place_image(anchor_images[2], 95, 95)
    canvas.save(args.save_path)
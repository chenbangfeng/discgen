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

from PIL import Image

channels = 4

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
        self.xspread_ratio = self.canvas_xspread / self.xspread
        self.yspread_ratio = self.canvas_yspread / self.yspread

    # To map
    # [A, B] --> [a, b]
    # use this formula
    # (val - A)*(b-a)/(B-A) + a
    # A,B is virtual
    # a,b is canvas
    def map_to_canvas(self, x, y):
        new_x = (x - self.xmin) * self.xspread_ratio + self.canvas_xmin
        new_y = (y - self.ymin) * self.yspread_ratio + self.canvas_ymin
        return new_x, new_y

    def place_square(self, x, y):
        square = np.zeros((channels, 64, 64))
        square.fill(1)
        cx, cy = self.map_to_canvas(x, y)
        self.pixels[:, (cy-32):(cy+32), (cx-32):(cx+32)] = square

    def save(self, save_path):
        out = np.dstack(self.pixels)
        out = (255 * out).astype(np.uint8)
        img = Image.fromarray(out)
        img.save(save_path)

# def create_canvas(width, height):
#     "canvas is just an array"
#     canvas = np.zeros((channels, height, width))
#     return canvas

def place_white_square(canvas, x, y):
    square = np.zeros((channels, 64, 64))
    square.fill(1)
    canvas[:, (y-32):(y+32), (x-32):(x+32)] = square

def save_canvas(canvas, save_path):
    out = np.dstack(canvas)
    out = (255 * out).astype(np.uint8)
    img = Image.fromarray(out)
    img.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--width", type=int, default=256,
                        help="width of canvas to render in pixels")
    parser.add_argument("--height", type=int, default=256,
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
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # canvas = create_canvas(args.width, args.height)
    # place_white_square(canvas, 50, 50)
    # save_canvas(canvas, args.save_path)
    canvas = Canvas(args.width, args.height, args.xmin, args.xmax, args.ymin, args.ymax)
    canvas.place_square(25, 25)
    canvas.place_square(50, 50)
    canvas.place_square(75, 75)
    canvas.save(args.save_path)
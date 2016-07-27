##TODO: OVERKILL
"""Does facegrid stuff."""
import argparse

import idx2numpy
from matplotlib import pylab as plt
import numpy as np
import os
import csv
from scipy.misc import imread
from scipy.misc import imresize 

import tarfile
import csv
import gzip
import shutil
from random import shuffle
import zipfile
import json
import sys

from PIL import Image

from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path
from annoy import AnnoyIndex

def json_list_to_array(json_list):
    files = json_list.split(",")
    encoded = []
    for file in files:
        with open(file) as json_file:
            encoded = encoded + json.load(json_file)
    return np.array(encoded)

def get_data_sources(dataset, split):
    # sources = ('features', 'targets') if include_targets else ('features',)
    if split == "all":
        splits = ('train', 'valid', 'test')
    elif split == "nontrain":
        splits = ('valid', 'test')
    else:
        splits = (split,)

    dataset_fname = find_in_data_path("{}.hdf5".format(dataset))
    split_sets = H5PYDataset(dataset_fname, which_sets=splits, sources=['features', 'targets', 'indexes'])
    print split_sets.num_examples, split_sets.provides_sources
    handle = split_sets.open()
    split_data = split_sets.get_data(handle, slice(0, split_sets.num_examples))
    split_sets.close(handle)
    print split_data[0].shape
    print split_data[1].shape
    print split_data[2].shape
    return split_data

def build_annoy_index(encoded, outfile):
    input_shape = encoded.shape
    f = input_shape[1]
    t = AnnoyIndex(f, metric='angular')  # Length of item vector that will be indexed
    for i,v in enumerate(encoded):
        t.add_item(i, v)

    t.build(100) # 10 trees
    if outfile is not None:
        t.save(outfile)

    return t

def load_annoy_index(infile, z_dim):
    t = AnnoyIndex(z_dim)
    t.load(infile) # super fast, will just mmap the file
    return t

def neighbors_to_grid(neighbors, imdata, gsize, with_center=False):
    canvas = np.zeros((gsize*3, gsize*5, 3)).astype(np.uint8)
    if with_center:
        offsets = [ [0, 0] ]
    else:
        offsets = []
    offsets += [
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
        [0, -2],
        [0, 2],
        [-1, -2],
        [1, -2],
        [-1, 2],
        [1, 2]
    ]
    cx = gsize*2
    cy = gsize*1
    for i, offset in enumerate(offsets):
        n = neighbors[i]
        im = np.dstack(imdata[n]).astype(np.uint8)
        offy = cy + gsize*offset[0]
        offx = cx + gsize*offset[1]
        canvas[offy:offy+gsize, offx:offx+gsize, :] = im
    return Image.fromarray(canvas)

def neighbors_to_rfgrid(neighbors, encoded, imdata, gsize):

    from tsne import bh_sne
    canvas = np.zeros((gsize*3, gsize*5, 3)).astype(np.uint8)

    vectors_list = []
    for n in neighbors:
        vectors_list.append(encoded[n])
    vectors = np.array(vectors_list)
    xy = bh_sne(vectors, perplexity=4., theta=0)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_bgcolor('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.autoscale_view(True,True,True)
    ax.invert_yaxis()
    ax.scatter(xy[:,0],xy[:,1],  edgecolors='none',marker='s',s=7.5)  # , c = vectors[:,:3]
    plt.savefig("plot.png")

    from rasterfairy import rasterfairy
    grid_xy, quadrants = rasterfairy.transformPointCloud2D(xy,target=(5,3))
    indices = []
    for i in range(15):
        indices.append(quadrants[i]["indices"][0])

    i = 0
    for cur_y in range(3):
        for cur_x in range(5):
            cur_index = indices[i]
            n = neighbors[cur_index]
            im = np.dstack(imdata[n]).astype(np.uint8)
            offy = gsize * cur_y
            offx = gsize * cur_x
            canvas[offy:offy+gsize, offx:offx+gsize, :] = im
            i = i + 1

    return Image.fromarray(canvas)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument('--build_annoy', dest='build_annoy',
                        default=False, action='store_true')
    parser.add_argument("--jsons", type=str, default=None,
                        help="Comma separated list of json arrays")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Source dataset.")
    parser.add_argument('--annoy_index', dest='annoy_index', default=None,
                        help="Annoy index.")
    parser.add_argument('--split', dest='split', default="all",
                        help="Which split to use from the dataset (train/nontrain/valid/test/any).")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument("--z-dim", dest='z_dim', type=int, default=100,
                        help="z dimension")
    parser.add_argument('--outdir', dest='outdir', default="neighborgrids",
                        help="Output dir for neighborgrids.")
    parser.add_argument('--range', dest='range', default="0,100",
                        help="Range of indexes to run.")
    args = parser.parse_args()

    encoded = json_list_to_array(args.jsons)
    print(encoded.shape)
    if args.build_annoy:
        aindex = build_annoy_index(encoded, args.annoy_index)
        sys.exit(0)

    # open annoy index and spit out some neighborgrids
    aindex = load_annoy_index(args.annoy_index, args.z_dim)
    data = get_data_sources(args.dataset, args.split)
    _, _, image_size, _ = data[0].shape
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    r = map(int, args.range.split(","))
    if len(r) == 1:
        r = [r[0], r[0]+1]
    for i in range(r[0], r[1]):
        neighbors = aindex.get_nns_by_item(i, 15, include_distances=True) # will find the 20 nearest neighbors
        # g = neighbors_to_grid(neighbors[0], data[0], image_size, with_center=True)
        g = neighbors_to_rfgrid(neighbors[0], encoded, data[0], image_size)
        g.save("{}/index_{:03d}.png".format(args.outdir, i))

# 'encodings/celeba_dlib_128_200z_a02.annoy'
#!/usr/bin/env python

"""Routines for accessing fuel datasets."""
import numpy as np
from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from discgen.utils import Colorize

def get_dataset_iterator(dataset, split, include_targets=False, unit_scale=True):
    """Get iterator for dataset, split, targets (labels) and scaling (from 255 to 1.0)"""
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
    if unit_scale:
        datastream.default_transformers = uint8_pixels_to_floatX(('features',))

    train_stream = DataStream.default_stream(
        dataset=datastream,
        iteration_scheme=SequentialExampleScheme(datastream.num_examples))

    it = train_stream.get_epoch_iterator()
    return it

# get images from dataset. numanchors=None to get all. image_size only needed for color conversion
def get_anchor_images(dataset, split, offset=0, stepsize=1, numanchors=150, allowed=None, prohibited=None, image_size=64, color_convert=False, include_targets=True, unit_scale=True):
    """Get images in np array with filters"""
    it = get_dataset_iterator(dataset, split, include_targets, unit_scale=unit_scale)

    anchors = []
    for i in range(offset):
        cur = it.next()
    try:
        while numanchors == None or len(anchors) < numanchors:
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
    except StopIteration:
        if numanchors is not None:
            print("Warning: only read {} of {} requested anchor images".format(len(anchors), numanchors))

    return np.array(anchors)

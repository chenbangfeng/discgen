"""Trains a CelebA-like classifier with up to 64 attributes."""
# TODO: cleanup imports (most of this not needed)

import argparse
import logging

import numpy as np
from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import Rectifier, Logistic
from blocks.bricks.bn import SpatialBatchNormalization, BatchNormalizedMLP
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import (ComputationGraph, apply_batch_normalization,
                          get_batch_normalization_updates, apply_dropout)
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.serialization import load
from blocks.roles import OUTPUT
from theano import tensor

from plat.fuel_helper import Colorize

import theano
from blocks.select import Selector
from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers.defaults import uint8_pixels_to_floatX
from itertools import islice

def get_all_data_inorder(filename, batch_size):
    sources = ('features', 'targets')

    dataset_fname = find_in_data_path(filename+'.hdf5')
    data_all = H5PYDataset(dataset_fname, which_sets=['train', 'valid', 'test'],
                             sources=sources)
    data_all.default_transformers = uint8_pixels_to_floatX(('features',))
    main_stream = DataStream.default_stream(
        dataset=data_all,
        iteration_scheme=SequentialScheme(data_all.num_examples, batch_size))
    color_stream = Colorize(main_stream, which_sources=('features',))
    return data_all.num_examples, color_stream


def create_running_graphs(classifier):
    try:
        classifier_model = Model(load(classifier).algorithm.cost)
    except AttributeError:
        # newer version of blocks
        with open(classifier, 'rb') as src:
            classifier_model = Model(load(src).algorithm.cost)

    selector = Selector(classifier_model.top_bricks)
    convnet, = selector.select('/convnet').bricks
    mlp, = selector.select('/mlp').bricks

    x = tensor.tensor4('features')
    y_hat = mlp.apply(convnet.apply(x).flatten(ndim=2))
    cg = ComputationGraph([y_hat])
    return cg


def run(batch_size, classifier, dataset):
    print('Opening data stream...')
    numdata, datastream = get_all_data_inorder(filename=dataset, batch_size=batch_size)
    print('Loaded {} items in stream.'.format(numdata))
    print('Loading saved model...')
    computation_graph = create_running_graphs(classifier)

    print('Compiling reconstruction function...')
    reconstruction_function = theano.function(
        computation_graph.inputs, computation_graph.outputs)

    it = datastream.get_epoch_iterator()
    for i in range(numdata):
        n = next(it)
        yhat = reconstruction_function(n[0])

        yn = np.array(yhat[0])
        nn = np.array(n[1])

        # print(yn.shape)
        # print(nn.shape)
        # print(yn)
        # print(nn)
        ym = np.argmax(yn, axis=1)
        nm = np.argmax(nn, axis=1)
        # print(ym)
        # print(nm)
        d = ym - nm
        # print(d)
        f = np.count_nonzero(d)
        print("RESULT,{},{}".format(i,f))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Run a classifier and print results")
    parser.add_argument('--classifier', dest='classifier', type=str,
                        default="celeba_classifier.zip")
    parser.add_argument("--batch-size", type=int, dest="batch_size",
                        default=62, help="Size of each mini-batch")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Use a different dataset for training.")

    args = parser.parse_args()
    allowed = None
    run(args.batch_size, args.classifier, args.dataset)

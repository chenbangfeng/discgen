"""Trains a CelebA-like classifier with up to 64 attributes."""
# TODO: cleanup imports (most of this not needed)

import argparse
import logging

# TODO: cleanup imports (most of this not needed)
import numpy
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

def create_model_bricks(image_size, depth):
    # original celebA64 was depth=3 (went to bach_norm6)
    layers = []
    if(depth > 0):
        layers = layers + [
            Convolutional(
                filter_size=(4, 4),
                num_filters=32,
                name='conv1'),
            SpatialBatchNormalization(name='batch_norm1'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=32,
                name='conv2'),
            SpatialBatchNormalization(name='batch_norm2'),
            Rectifier(),
        ]
    if(depth > 1):
        layers = layers + [
            Convolutional(
                filter_size=(4, 4),
                num_filters=64,
                name='conv3'),
            SpatialBatchNormalization(name='batch_norm3'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=64,
                name='conv4'),
            SpatialBatchNormalization(name='batch_norm4'),
            Rectifier(),
        ]
    if(depth > 2):
        layers = layers + [
            Convolutional(
                filter_size=(3, 3),
                num_filters=128,
                name='conv5'),
            SpatialBatchNormalization(name='batch_norm5'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=128,
                name='conv6'),
            SpatialBatchNormalization(name='batch_norm6'),
            Rectifier(),
        ]
    if(depth > 3):
        layers = layers + [
            Convolutional(
                filter_size=(3, 3),
                num_filters=256,
                name='conv7'),
            SpatialBatchNormalization(name='batch_norm7'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=256,
                name='conv8'),
            SpatialBatchNormalization(name='batch_norm8'),
            Rectifier(),
        ]
    if(depth > 4):
        layers = layers + [
            Convolutional(
                filter_size=(3, 3),
                num_filters=512,
                name='conv9'),
            SpatialBatchNormalization(name='batch_norm9'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=512,
                name='conv10'),
            SpatialBatchNormalization(name='batch_norm10'),
            Rectifier(),
        ]
    if(depth > 5):
        layers = layers + [
            Convolutional(
                filter_size=(3, 3),
                num_filters=512,
                name='conv11'),
            SpatialBatchNormalization(name='batch_norm11'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=512,
                name='conv12'),
            SpatialBatchNormalization(name='batch_norm12'),
            Rectifier(),
        ]
    if(depth > 6):
        layers = layers + [
            Convolutional(
                filter_size=(3, 3),
                num_filters=512,
                name='conv13'),
            SpatialBatchNormalization(name='batch_norm13'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=512,
                name='conv14'),
            SpatialBatchNormalization(name='batch_norm14'),
            Rectifier(),
        ]
    if(depth > 7):
        layers = layers + [
            Convolutional(
                filter_size=(3, 3),
                num_filters=512,
                name='conv15'),
            SpatialBatchNormalization(name='batch_norm15'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=512,
                name='conv16'),
            SpatialBatchNormalization(name='batch_norm16'),
            Rectifier(),
        ]

    print("creating model of depth {} with {} layers".format(depth, len(layers)))

    convnet = ConvolutionalSequence(
        layers=layers,
        num_channels=3,
        image_size=(image_size, image_size),
        use_bias=False,
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        name='convnet')
    convnet.initialize()

    mlp = BatchNormalizedMLP(
        activations=[Rectifier(), Logistic()],
        dims=[numpy.prod(convnet.get_dim('output')), 1000, 64],
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        name='mlp')
    mlp.initialize()

    return convnet, mlp, len(layers)


def create_training_computation_graphs(image_size, net_depth):
    x = tensor.tensor4('features')
    y = tensor.imatrix('targets')

    convnet, mlp, num_conv_layers = create_model_bricks(image_size=image_size, depth=net_depth)
    y_hat = mlp.apply(convnet.apply(x).flatten(ndim=2))
    cost = BinaryCrossEntropy().apply(y, y_hat)
    accuracy = 1 - tensor.neq(y > 0.5, y_hat > 0.5).mean()
    cg = ComputationGraph([cost, accuracy])

    # Create a graph which uses batch statistics for batch normalization
    # as well as dropout on selected variables
    bn_cg = apply_batch_normalization(cg)
    drop_layers = range(5, num_conv_layers, 6)
    print("Applying drop to layers: {}".format(drop_layers))
    bricks_to_drop = ([convnet.layers[i] for i in drop_layers] +
                      [mlp.application_methods[1].brick])
    variables_to_drop = VariableFilter(
        roles=[OUTPUT], bricks=bricks_to_drop)(bn_cg.variables)
    bn_dropout_cg = apply_dropout(bn_cg, variables_to_drop, 0.5)

    return cg, bn_dropout_cg

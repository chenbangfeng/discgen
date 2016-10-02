"""Trains a VAE on any fuel 64x64 image dataset."""
# TODO: cleanup imports (most of this not needed)

import argparse
import logging

import sys
import numpy
import theano
from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import Sequence, Random, Rectifier, Identity, MLP, Logistic
from blocks.bricks.bn import (BatchNormalization, BatchNormalizedMLP,
                              SpatialBatchNormalization)
from blocks.bricks.conv import (Convolutional, ConvolutionalTranspose,
                                ConvolutionalSequence)
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import (ComputationGraph, get_batch_normalization_updates,
                          batch_normalization)
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import add_role, OUTPUT, PARAMETER
from blocks.select import Selector
from blocks.serialization import load
from blocks.utils import find_bricks, shared_floatx
from theano import tensor

from discgen.utils import create_celeba_streams
from discgen.interface import DiscGenModel


def create_model_bricks(z_dim, image_size, depth):

    g_image_size = image_size
    g_image_size2 = g_image_size/2
    g_image_size3 = g_image_size/4
    g_image_size4 = g_image_size/8
    g_image_size5 = g_image_size/16

    encoder_layers = []
    if depth > 0:
        encoder_layers = encoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=32,
                name='conv1'),
            SpatialBatchNormalization(name='batch_norm1'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=32,
                name='conv2'),
            SpatialBatchNormalization(name='batch_norm2'),
            Rectifier(),
            Convolutional(
                filter_size=(2, 2),
                step=(2, 2),
                num_filters=32,
                name='conv3'),
            SpatialBatchNormalization(name='batch_norm3'),
            Rectifier()
        ]
    if depth > 1:
        encoder_layers = encoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=64,
                name='conv4'),
            SpatialBatchNormalization(name='batch_norm4'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=64,
                name='conv5'),
            SpatialBatchNormalization(name='batch_norm5'),
            Rectifier(),
            Convolutional(
                filter_size=(2, 2),
                step=(2, 2),
                num_filters=64,
                name='conv6'),
            SpatialBatchNormalization(name='batch_norm6'),
            Rectifier()
        ]
    if depth > 2:
        encoder_layers = encoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=128,
                name='conv7'),
            SpatialBatchNormalization(name='batch_norm7'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=128,
                name='conv8'),
            SpatialBatchNormalization(name='batch_norm8'),
            Rectifier(),
            Convolutional(
                filter_size=(2, 2),
                step=(2, 2),
                num_filters=128,
                name='conv9'),
            SpatialBatchNormalization(name='batch_norm9'),
            Rectifier()
        ]
    if depth > 3:
        encoder_layers = encoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=256,
                name='conv10'),
            SpatialBatchNormalization(name='batch_norm10'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=256,
                name='conv11'),
            SpatialBatchNormalization(name='batch_norm11'),
            Rectifier(),
            Convolutional(
                filter_size=(2, 2),
                step=(2, 2),
                num_filters=256,
                name='conv12'),
            SpatialBatchNormalization(name='batch_norm12'),
            Rectifier(),
        ]
    if depth > 4:
        encoder_layers = encoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv13'),
            SpatialBatchNormalization(name='batch_norm13'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv14'),
            SpatialBatchNormalization(name='batch_norm14'),
            Rectifier(),
            Convolutional(
                filter_size=(2, 2),
                step=(2, 2),
                num_filters=512,
                name='conv15'),
            SpatialBatchNormalization(name='batch_norm15'),
            Rectifier()
        ]

    decoder_layers = []
    if depth > 4:
        decoder_layers = decoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv_n3'),
            SpatialBatchNormalization(name='batch_norm_n3'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=512,
                name='conv_n2'),
            SpatialBatchNormalization(name='batch_norm_n2'),
            Rectifier(),
            ConvolutionalTranspose(
                filter_size=(2, 2),
                step=(2, 2),
                original_image_size=(g_image_size5, g_image_size5),
                num_filters=512,
                name='conv_n1'),
            SpatialBatchNormalization(name='batch_norm_n1'),
            Rectifier()
        ]

    if depth > 3:
        decoder_layers = decoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=256,
                name='conv1'),
            SpatialBatchNormalization(name='batch_norm1'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=256,
                name='conv2'),
            SpatialBatchNormalization(name='batch_norm2'),
            Rectifier(),
            ConvolutionalTranspose(
                filter_size=(2, 2),
                step=(2, 2),
                original_image_size=(g_image_size4, g_image_size4),
                num_filters=256,
                name='conv3'),
            SpatialBatchNormalization(name='batch_norm3'),
            Rectifier()
        ]

    if depth > 2:
        decoder_layers = decoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=128,
                name='conv4'),
            SpatialBatchNormalization(name='batch_norm4'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=128,
                name='conv5'),
            SpatialBatchNormalization(name='batch_norm5'),
            Rectifier(),
            ConvolutionalTranspose(
                filter_size=(2, 2),
                step=(2, 2),
                original_image_size=(g_image_size3, g_image_size3),
                num_filters=128,
                name='conv6'),
            SpatialBatchNormalization(name='batch_norm6'),
            Rectifier()
        ]

    if depth > 1:
        decoder_layers = decoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=64,
                name='conv7'),
            SpatialBatchNormalization(name='batch_norm7'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=64,
                name='conv8'),
            SpatialBatchNormalization(name='batch_norm8'),
            Rectifier(),
            ConvolutionalTranspose(
                filter_size=(2, 2),
                step=(2, 2),
                original_image_size=(g_image_size2, g_image_size2),
                num_filters=64,
                name='conv9'),
            SpatialBatchNormalization(name='batch_norm9'),
            Rectifier()
        ]

    if depth > 0:
        decoder_layers = decoder_layers + [
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=32,
                name='conv10'),
            SpatialBatchNormalization(name='batch_norm10'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=32,
                name='conv11'),
            SpatialBatchNormalization(name='batch_norm11'),
            Rectifier(),
            ConvolutionalTranspose(
                filter_size=(2, 2),
                step=(2, 2),
                original_image_size=(g_image_size, g_image_size),
                num_filters=32,
                name='conv12'),
            SpatialBatchNormalization(name='batch_norm12'),
            Rectifier()
        ]

    decoder_layers = decoder_layers + [
        Convolutional(
            filter_size=(1, 1),
            num_filters=3,
            name='conv_out'),
        Logistic()
    ]
 
    print("creating model of depth {} with {} encoder and {} decoder layers".format(depth, len(encoder_layers), len(decoder_layers)))

    encoder_convnet = ConvolutionalSequence(
        layers=encoder_layers,
        num_channels=3,
        image_size=(g_image_size, g_image_size),
        use_bias=False,
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        name='encoder_convnet')
    encoder_convnet.initialize()

    encoder_filters = numpy.prod(encoder_convnet.get_dim('output'))

    encoder_mlp = MLP(
        dims=[encoder_filters, 1000, z_dim],
        activations=[Sequence([BatchNormalization(1000).apply,
                               Rectifier().apply], name='activation1'),
                     Identity().apply],
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        name='encoder_mlp')
    encoder_mlp.initialize()

    decoder_mlp = BatchNormalizedMLP(
        activations=[Rectifier(), Rectifier()],
        dims=[encoder_mlp.output_dim // 2, 1000, encoder_filters],
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        name='decoder_mlp')
    decoder_mlp.initialize()

    decoder_convnet = ConvolutionalSequence(
        layers=decoder_layers,
        num_channels=encoder_convnet.get_dim('output')[0],
        image_size=encoder_convnet.get_dim('output')[1:],
        use_bias=False,
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        name='decoder_convnet')
    decoder_convnet.initialize()

    return encoder_convnet, encoder_mlp, decoder_convnet, decoder_mlp


def create_training_computation_graphs(z_dim, image_size, net_depth, discriminative_regularization,
                                       classifer, vintage, reconstruction_factor,
                                       kl_factor, discriminative_factor, disc_weights):
    x = tensor.tensor4('features')
    pi = numpy.cast[theano.config.floatX](numpy.pi)

    bricks = create_model_bricks(z_dim=z_dim, image_size=image_size, depth=net_depth)
    encoder_convnet, encoder_mlp, decoder_convnet, decoder_mlp = bricks
    if discriminative_regularization:
        if vintage:
            classifier_model = Model(load(classifer).algorithm.cost)
        else:
            with open(classifer, 'rb') as src:
                classifier_model = Model(load(src).algorithm.cost)
        selector = Selector(classifier_model.top_bricks)
        classifier_convnet, = selector.select('/convnet').bricks
        classifier_mlp, = selector.select('/mlp').bricks

    random_brick = Random()

    # Initialize conditional variances
    log_sigma_theta = shared_floatx(
        numpy.zeros((3, image_size, image_size)), name='log_sigma_theta')
    add_role(log_sigma_theta, PARAMETER)
    variance_parameters = [log_sigma_theta]
    num_disc_layers = 0
    if discriminative_regularization:
        # We add discriminative regularization for the batch-normalized output
        # of the strided layers of the classifier.
        for layer in classifier_convnet.layers[1::3]:
            log_sigma = shared_floatx(
                numpy.zeros(layer.get_dim('output')),
                name='{}_log_sigma'.format(layer.name))
            add_role(log_sigma, PARAMETER)
            variance_parameters.append(log_sigma)
        # include mlp
        # DISABLED
        # log_sigma = shared_floatx(
        #     numpy.zeros([classifier_mlp.output_dim]),
        #     name='{}_log_sigma'.format("MLP"))
        # add_role(log_sigma, PARAMETER)
        # variance_parameters.append(log_sigma)
        # diagnostic
        num_disc_layers = len(variance_parameters)-1
        print("Applying discriminative regularization on {} layers".format(num_disc_layers))

    # Computation graph creation is encapsulated within this function in order
    # to allow selecting which parts of the graph will use batch statistics for
    # batch normalization and which parts will use population statistics.
    # Specifically, we'd like to use population statistics for the classifier
    # even in the training graph.
    def create_computation_graph():
        # Encode
        phi = encoder_mlp.apply(encoder_convnet.apply(x).flatten(ndim=2))
        nlat = encoder_mlp.output_dim // 2
        mu_phi = phi[:, :nlat]
        log_sigma_phi = phi[:, nlat:]
        # Sample from the approximate posterior
        epsilon = random_brick.theano_rng.normal(
            size=mu_phi.shape, dtype=mu_phi.dtype)
        z = mu_phi + epsilon * tensor.exp(log_sigma_phi)
        # Decode
        mu_theta = decoder_convnet.apply(
            decoder_mlp.apply(z).reshape(
                (-1,) + decoder_convnet.get_dim('input_')))
        log_sigma = log_sigma_theta.dimshuffle('x', 0, 1, 2)

        # Compute KL and reconstruction terms
        kl_term = 0.5 * (
            tensor.exp(2 * log_sigma_phi) + mu_phi ** 2 - 2 * log_sigma_phi - 1
        ).sum(axis=1)

        reconstruction_term = -0.5 * (
            tensor.log(2 * pi) + 2 * log_sigma +
            (x - mu_theta) ** 2 / tensor.exp(2 * log_sigma)
        ).sum(axis=[1, 2, 3])

        discriminative_layer_terms = [None] * num_disc_layers
        for i in range(num_disc_layers):
            discriminative_layer_terms[i] = tensor.zeros_like(kl_term)
        discriminative_term  = tensor.zeros_like(kl_term)
        if discriminative_regularization:
            # Propagate both the input and the reconstruction through the classifier
            acts_cg = ComputationGraph([classifier_mlp.apply(classifier_convnet.apply(x).flatten(ndim=2))])
            acts_hat_cg = ComputationGraph(
                [classifier_mlp.apply(classifier_convnet.apply(mu_theta).flatten(ndim=2))])

            # Retrieve activations of interest and compute discriminative
            # regularization reconstruction terms
            cur_layer = 0
            # CLASSIFIER MLP DISABLED
            # for i, zip_pair in enumerate(zip(classifier_convnet.layers[1::3] + [classifier_mlp],
            for i, zip_pair in enumerate(zip(classifier_convnet.layers[1::3],
                                        variance_parameters[1:])):

                layer, log_sigma = zip_pair
                variable_filter = VariableFilter(roles=[OUTPUT],
                                                 bricks=[layer])

                d, = variable_filter(acts_cg)
                d_hat, = variable_filter(acts_hat_cg)

                # TODO: this conditional could be less brittle
                if "mlp" in layer.name.lower():
                    log_sigma = log_sigma.dimshuffle('x', 0)
                    sumaxis = [1]
                else:
                    log_sigma = log_sigma.dimshuffle('x', 0, 1, 2)
                    sumaxis = [1, 2, 3]

                discriminative_layer_term_unweighted = -0.5 * (
                    tensor.log(2 * pi) + 2 * log_sigma +
                    (d - d_hat) ** 2 / tensor.exp(2 * log_sigma)
                ).sum(axis=sumaxis)

                discriminative_layer_terms[i] = discriminative_factor * disc_weights[cur_layer] * discriminative_layer_term_unweighted
                discriminative_term = discriminative_term + discriminative_layer_terms[i]

                cur_layer = cur_layer + 1

        # scale terms (disc is prescaled by layer)
        reconstruction_term = reconstruction_factor * reconstruction_term
        kl_term = kl_factor * kl_term

        # total_reconstruction_term is reconstruction + discriminative
        total_reconstruction_term = reconstruction_term + discriminative_term

        # cost is mean(kl - total reconstruction)
        cost = (kl_term - total_reconstruction_term).mean()

        return ComputationGraph([cost, kl_term,
                                 reconstruction_term, discriminative_term] + discriminative_layer_terms)

    cg = create_computation_graph()
    with batch_normalization(encoder_convnet, encoder_mlp,
                             decoder_convnet, decoder_mlp):
        bn_cg = create_computation_graph()

    return cg, bn_cg, variance_parameters

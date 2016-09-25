"""Trains a VAE on any fuel 64x64 image dataset."""
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
from chips.fuel_helper import create_custom_streams
from discgen.interface import DiscGenModel
from chips.samplecheckpoint import SampleCheckpoint


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


def run(batch_size, save_path, z_dim, oldmodel, discriminative_regularization,
        classifier, vintage, monitor_every, monitor_before, checkpoint_every, dataset, color_convert,
        image_size, net_depth, subdir,
        reconstruction_factor, kl_factor, discriminative_factor, disc_weights):

    if dataset:
        streams = create_custom_streams(filename=dataset,
                                        training_batch_size=batch_size,
                                        monitoring_batch_size=batch_size,
                                        include_targets=False,
                                        color_convert=color_convert)
    else:
        streams = create_celeba_streams(training_batch_size=batch_size,
                                        monitoring_batch_size=batch_size,
                                        include_targets=False)

    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams[:3]

    # Compute parameter updates for the batch normalization population
    # statistics. They are updated following an exponential moving average.
    rval = create_training_computation_graphs(
                z_dim, image_size, net_depth, discriminative_regularization, classifier,
                vintage, reconstruction_factor, kl_factor, discriminative_factor, disc_weights)
    cg, bn_cg, variance_parameters = rval

    pop_updates = list(
        set(get_batch_normalization_updates(bn_cg, allow_duplicates=True)))
    decay_rate = 0.05
    extra_updates = [(p, m * decay_rate + p * (1 - decay_rate))
                     for p, m in pop_updates]

    model = Model(bn_cg.outputs[0])

    selector = Selector(
        find_bricks(
            model.top_bricks,
            lambda brick: brick.name in ('encoder_convnet', 'encoder_mlp',
                                         'decoder_convnet', 'decoder_mlp')))
    parameters = list(selector.get_parameters().values()) + variance_parameters

    # Prepare algorithm
    step_rule = Adam()
    algorithm = GradientDescent(cost=bn_cg.outputs[0],
                                parameters=parameters,
                                step_rule=step_rule)
    algorithm.add_updates(extra_updates)

    # Prepare monitoring
    sys.setrecursionlimit(1000000)

    monitored_quantities_list = []
    for graph in [bn_cg, cg]:
        # cost, kl_term, reconstruction_term, discriminative_term = graph.outputs
        cost, kl_term, reconstruction_term, discriminative_term = graph.outputs[:4]
        discriminative_layer_terms = graph.outputs[4:]

        cost.name = 'nll_upper_bound'
        avg_kl_term = kl_term.mean(axis=0)
        avg_kl_term.name = 'avg_kl_term'
        avg_reconstruction_term = -reconstruction_term.mean(axis=0)
        avg_reconstruction_term.name = 'avg_reconstruction_term'
        avg_discriminative_term = discriminative_term.mean(axis=0)
        avg_discriminative_term.name = 'avg_discriminative_term'

        num_layer_terms = len(discriminative_layer_terms)
        avg_discriminative_layer_terms = [None] * num_layer_terms
        for i, term in enumerate(discriminative_layer_terms):
            avg_discriminative_layer_terms[i] = discriminative_layer_terms[i].mean(axis=0)
            avg_discriminative_layer_terms[i].name = "avg_discriminative_term_layer_{:02d}".format(i)

        monitored_quantities_list.append(
            [cost, avg_kl_term, avg_reconstruction_term,
             avg_discriminative_term] + avg_discriminative_layer_terms)

    train_monitoring = DataStreamMonitoring(
        monitored_quantities_list[0], train_monitor_stream, prefix="train",
        updates=extra_updates, after_epoch=False, before_first_epoch=True,
        every_n_epochs=monitor_every)
    valid_monitoring = DataStreamMonitoring(
        monitored_quantities_list[1], valid_monitor_stream, prefix="valid",
        after_epoch=False, before_first_epoch=monitor_before,
        every_n_epochs=monitor_every)

    # Prepare checkpoint
    checkpoint = Checkpoint(save_path, every_n_epochs=checkpoint_every,
                            before_training=True, use_cpickle=True)

    # TODO: why does z_dim=foo become foo/2?
    extensions = [Timing(), FinishAfter(after_n_epochs=100), checkpoint,
                  train_monitoring, valid_monitoring, 
                  SampleCheckpoint(interface=DiscGenModel, z_dim=z_dim/2, image_size=(image_size, image_size), channels=3, dataset=dataset, split="valid", save_subdir=subdir, before_training=True, after_epoch=True),
                  Printing(), ProgressBar()]
    main_loop = MainLoop(model=model, data_stream=main_loop_stream,
                         algorithm=algorithm, extensions=extensions)

    if oldmodel is not None:
        print("Initializing parameters with old model {}".format(oldmodel))
        try:
            saved_model = load(oldmodel)
        except AttributeError:
            # newer version of blocks
            with open(oldmodel, 'rb') as src:
                saved_model = load(src)
        main_loop.model.set_parameter_values(
            saved_model.model.get_parameter_values())
        del saved_model

    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Train a VAE on a fuel dataset")
    parser.add_argument("--regularize", action='store_true',
                        help="apply discriminative regularization")
    parser.add_argument('--classifier', dest='classifier', type=str,
                        default="celeba_classifier.zip")
    parser.add_argument('--vintage', dest='vintage',
                        default=False, action='store_true',
                        help="Are you running a vintage version of blocks?")
    parser.add_argument('--model', dest='model', type=str,
                        default="celeba_vae_regularization.zip")
    parser.add_argument("--batch-size", type=int, dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                        default=100, help="Z-vector dimension")
    parser.add_argument("--reconstruction-factor", type=float,
                        dest="reconstruction_factor", default=1.0,
                        help="Scaling Factor for reconstruction term")
    parser.add_argument("--kl-factor", type=float, dest="kl_factor",
                        default=1.0, help="Scaling Factor for KL term")
    parser.add_argument("--discriminative-factor", type=float,
                        dest="discriminative_factor", default=1.0,
                        help="Scaling Factor for discriminative term")
    parser.add_argument("--discriminative-layer-weights", type=str,
                        dest="discriminative_layer_weights", default="1,0,1,0,1,0",
                        help="Weights for each of 6 discriminitive layers")
    parser.add_argument("--monitor-every", type=int, dest="monitor_every",
                        default=5, help="Frequency in epochs for monitoring")
    parser.add_argument("--checkpoint-every", type=int,
                        dest="checkpoint_every", default=5,
                        help="Frequency in epochs for checkpointing")
    parser.add_argument('--monitor-before', dest='monitor_before',
                        default=False, action='store_true',
                        help="monitor at epoch 0")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Dataset for training.")
    parser.add_argument('--color-convert', dest='color_convert',
                        default=False, action='store_true',
                        help="Convert source dataset to color from grayscale.")
    parser.add_argument("--oldmodel", type=str, default=None,
                        help="Use a model file created by a previous run as\
                        a starting point for parameters")
    parser.add_argument("--subdir", dest='subdir', type=str, default="output",
                        help="Subdirectory for output files (images)")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument("--net-depth", dest='net_depth', type=int, default=5,
                        help="network depth from 1-5")
    args = parser.parse_args()
    disc_weights = map(float, args.discriminative_layer_weights.split(","))
    run(args.batch_size, args.model, args.z_dim, args.oldmodel,
        args.regularize, args.classifier, args.vintage, args.monitor_every, args.monitor_before,
        args.checkpoint_every, args.dataset, args.color_convert,
        args.image_size, args.net_depth, args.subdir,
        args.reconstruction_factor, args.kl_factor, args.discriminative_factor, disc_weights)

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
from plat.fuel_helper import create_custom_streams
from discgen.interface import DiscGenModel
from plat.training.samplecheckpoint import SampleCheckpoint
from discgen.vae import create_training_computation_graphs

def run(batch_size, save_path, z_dim, oldmodel, discriminative_regularization,
        classifier, vintage, monitor_every, monitor_before, checkpoint_every, dataset, color_convert,
        image_size, net_depth, subdir,
        reconstruction_factor, kl_factor, discriminative_factor, disc_weights,
        num_epochs):

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
        updates=extra_updates, after_epoch=False, before_first_epoch=monitor_before,
        every_n_epochs=monitor_every)
    valid_monitoring = DataStreamMonitoring(
        monitored_quantities_list[1], valid_monitor_stream, prefix="valid",
        after_epoch=False, before_first_epoch=monitor_before,
        every_n_epochs=monitor_every)

    # Prepare checkpoint
    checkpoint = Checkpoint(save_path, every_n_epochs=checkpoint_every,
                            before_training=True, use_cpickle=True)

    sample_checkpoint = SampleCheckpoint(interface=DiscGenModel, z_dim=z_dim/2,
                            image_size=(image_size, image_size), channels=3,
                            dataset=dataset, split="valid", save_subdir=subdir,
                            before_training=True, after_epoch=True)
    # TODO: why does z_dim=foo become foo/2?
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  checkpoint,
                  sample_checkpoint,
                  train_monitoring, valid_monitoring, 
                  Printing(),
                  ProgressBar()]
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
    parser.add_argument("--num-epochs", type=int, dest="num_epochs",
                        default=100, help="Stop training after num-epochs.")
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
        args.reconstruction_factor, args.kl_factor, args.discriminative_factor, disc_weights,
        args.num_epochs)

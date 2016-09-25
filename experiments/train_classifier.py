"""Trains a CelebA-like classifier with up to 64 attributes."""
import argparse
import logging

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

from chips.fuel_helper import create_custom_streams

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


def run(batch_size, classifier, oldmodel, monitor_every, checkpoint_every, final_epoch,
        dataset, color_convert, image_size, net_depth, allowed, stretch):

    streams = create_custom_streams(filename=dataset,
                                    training_batch_size=batch_size,
                                    monitoring_batch_size=batch_size,
                                    include_targets=True,
                                    color_convert=color_convert,
                                    allowed=allowed,
                                    stretch=stretch)

    main_loop_stream = streams[0]
    train_monitor_stream = streams[1]
    valid_monitor_stream = streams[2]

    cg, bn_dropout_cg = create_training_computation_graphs(image_size, net_depth)

    model = Model(bn_dropout_cg.outputs[0])

    # Compute parameter updates for the batch normalization population
    # statistics. They are updated following an exponential moving average.
    pop_updates = get_batch_normalization_updates(bn_dropout_cg)
    decay_rate = 0.05
    extra_updates = [(p, m * decay_rate + p * (1 - decay_rate))
                     for p, m in pop_updates]

    # Prepare algorithm
    step_rule = Adam()
    algorithm = GradientDescent(cost=bn_dropout_cg.outputs[0],
                                parameters=bn_dropout_cg.parameters,
                                step_rule=step_rule)
    algorithm.add_updates(extra_updates)

    # Prepare monitoring
    cost = bn_dropout_cg.outputs[0]
    cost.name = 'cost'
    train_monitoring = DataStreamMonitoring(
        [cost], train_monitor_stream, prefix="train",
        before_first_epoch=False, after_epoch=False, after_training=True,
        updates=extra_updates)

    cost, accuracy = cg.outputs
    cost.name = 'cost'
    accuracy.name = 'accuracy'
    monitored_quantities = [cost, accuracy]
    valid_monitoring = DataStreamMonitoring(
        monitored_quantities, valid_monitor_stream, prefix="valid",
        before_first_epoch=True, after_epoch=False,
        every_n_epochs=monitor_every)

    # Prepare checkpoint
    checkpoint = Checkpoint(classifier, every_n_epochs=checkpoint_every,
                            use_cpickle=True)

    extensions = [Timing(), FinishAfter(after_n_epochs=final_epoch), train_monitoring,
                  valid_monitoring, checkpoint, Printing(), ProgressBar()]
    main_loop = MainLoop(model=model, data_stream=main_loop_stream,
                         algorithm=algorithm, extensions=extensions)

    if oldmodel is not None:
        print("Initializing parameters with old model {}".format(oldmodel))
        with open(oldmodel, 'rb') as src:
            saved_model = load(src)
            main_loop.model.set_parameter_values(
                saved_model.model.get_parameter_values())
            del saved_model

    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Train a classifier on the CelebA dataset")
    parser.add_argument('--classifier', dest='classifier', type=str,
                        default="celeba_classifier.zip")
    parser.add_argument("--allowed", dest='allowed', type=str, default=None,
                        help="Only allow whitelisted labels L1,L2,...")
    parser.add_argument("--stretch", dest='stretch',
                        default=False, action='store_true',
                        help="Stretch dataset labels to standard length")
    parser.add_argument("--batch-size", type=int, dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--final-epoch", type=int, dest="final_epoch",
                        default=50, help="Quit after which epoch")
    parser.add_argument("--monitor-every", type=int, dest="monitor_every",
                        default=5, help="Frequency in epochs for monitoring")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        dest="checkpoint_every",
                        help="Frequency in epochs for checkpointing")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Use a different dataset for training.")
    parser.add_argument('--color-convert', dest='color_convert',
                        default=False, action='store_true',
                        help="Convert source dataset to color from grayscale.")
    parser.add_argument("--oldmodel", type=str, default=None,
                        help="Load old model as starting point")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument("--net-depth", dest='net_depth', type=int, default=4,
                        help="network depth from 1-5")

    args = parser.parse_args()
    allowed = None
    if(args.allowed):
        allowed = map(int, args.allowed.split(","))
    run(args.batch_size, args.classifier, args.oldmodel, args.monitor_every,
        args.checkpoint_every, args.final_epoch, args.dataset, args.color_convert,
        args.image_size, args.net_depth,
        allowed, args.stretch)

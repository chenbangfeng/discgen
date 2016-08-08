import theano
from blocks.model import Model
from blocks.serialization import load
from blocks.select import Selector
from blocks.graph import ComputationGraph
from sample_utils import get_image_encoder_function
from blocks.utils import shared_floatx

class DiscGenModel:
    def __init__(self, filename=None, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = Model(load(filename).algorithm.cost)

    def encode_images(self, images):
        encoder_function = get_image_encoder_function(self.model)
        print('Encoding...')
        examples, latents = encoder_function(images)
        return latents

    def get_zdim(self):
        selector = Selector(self.model.top_bricks)
        decoder_mlp, = selector.select('/decoder_mlp').bricks
        return decoder_mlp.input_dim

    def sample_at(self, z):
        selector = Selector(self.model.top_bricks)
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


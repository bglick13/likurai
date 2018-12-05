from .. import floatX

from . import pm
from . import relu, softmax
from . import randn
from functools import partial


class Layer:
    def __init__(self, input_size, output_size):
        self.weights_init = randn(input_size, output_size).astype(floatX)
        self.bias_init    = randn(output_size).astype(floatX)


class BayesianDenseLayer(Layer):
    def __init__(self, name, input_size, output_size, activation: str or function = None):
        super().__init__(input_size, output_size)
        self.weights = pm.Normal('{}_weights'.format(name), mu=0., sd=1., shape=(input_size, output_size))
        self.bias    = pm.Normal('{}_bias'.format(name), mu=0., sd=1., shape=output_size)
        if isinstance(activation, str):
            if activation == 'relu':
                self.activation = relu
            elif activation == 'linear':
                self.activation = None
            elif activation == 'softmax':
                self.activation = softmax

    def __call__(self, *args, **kwargs):
        if self.activation is None:
            return pm.math.dot(args[0], self.weights) + self.bias_init
        return self.activation(pm.math.dot(args[0], self.weights) + self.bias_init)




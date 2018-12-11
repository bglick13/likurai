from .. import floatX

import pymc3 as pm
from theano.tensor import concatenate
import theano.tensor.nnet as nn
import theano.tensor as tt
from theano.tensor.signal.pool import pool_2d
import numpy as np


def retrieve_distribution(dist):
    return getattr(pm, dist)


def retrieve_activation(act):
    return getattr(nn, act)


class Layer:
    def __init__(self):
        self.name = None
        self.built = False
        self.input = None

    def __call__(self, *args, **kwargs):
        pass

    def build(self, *args):
        pass


class Likelihood(Layer):
    def __init__(self, dist: str, connected_param, **kwargs):
        """
        Creates rhe likelihood distribution that the model is conditioned on
        :param dist: str - the type of distribution the observed data follows
        :param connected_param: str - the name of the param the model output should parameterize
        :param kwargs:
        :return:
        """
        super().__init__()
        self.name = 'likelihood'
        self.dist = getattr(pm, dist)
        self.connected_param = connected_param

    def __call__(self, *args, **kwargs):
        if 'concat_axis' in kwargs:
            concat_axis = kwargs.pop('concat_axis')
        else:
            concat_axis = 1
        input = concatenate(args, axis=concat_axis)
        _params = {self.connected_param: input}

        try:
            jitter = kwargs.pop('jitter')
        except KeyError:
            jitter = 0

        _params[self.connected_param] += jitter
        for key, value in kwargs.items():
            if isinstance(value, dict):
                _dist = getattr(pm, value.pop('dist'))
                _params[key] = _dist(**value)
            elif isinstance(value, (float, int)):
                _params[key] = value
        observed = kwargs.pop('observed')
        total_size = len(observed.get_value())
        likelihood = self.dist('likelihood', observed=observed, total_size=total_size, **_params)


class BayesianDense(Layer):
    def __init__(self, name, neurons=None, input_size=None, activation: str or function=None, mu=0., sd=0.5, use_bias=True, **kwargs):
        """
        Initialize a basic dense layer in a Bayesian framework
        :param name: Name of the layer
        :param weight_dist: Type of distribution. Currently only supports ('normal', 'bernoulli')
        :param input_size: Input size of the layer
        :param output_size: Number of output neurons
        :param activation: Layer activation. Currently only supports ('relu', 'sigmoid', 'linear', 'softmax')
        :param use_bias: Whether to use bias nodes or not
        :param kwargs: Should be dict of dict containing 'weight_kwargs', 'bias_kwargs'
        """
        super().__init__()
        self.name = name
        self.neurons = neurons
        self.input_size = input_size
        self.use_bias = use_bias
        self.mu = mu
        self.sd = sd
        self.weights = None
        self.bias = None

        if isinstance(activation, str):
            if activation == 'linear' or activation is None:
                self.activation = None
            else:
                self.activation = retrieve_activation(activation)

    def build(self, *args):
        self.input = concatenate(args, axis=1)
        shape = (self.input_size, self.neurons)
        self.weights = pm.Normal('{}_weights'.format(self.name), mu=self.mu, sd=self.sd, shape=shape)
        if self.use_bias:
            self.bias = pm.Normal('{}_bias'.format(self.name), mu=0., sd=1., shape=self.neurons)
        self.built = True

    def __call__(self, *args, **kwargs):
        if not self.built:
            self.build(*args)
        act = pm.math.dot(self.input, self.weights)
        if self.use_bias:
            act = act + self.bias
        if self.activation is None:
            return act
        else:
            return self.activation(act)


class HierarchicalBayesianDense(Layer):
    def __init__(self, name, neurons=None, input_size=None, n_groups=None, mu=0., sd=.5,
        activation: str or function=None, use_bias=True, **kwargs):
        """
        Initialize a basic dense layer in a Bayesian framework
        :param name: Name of the layer
        :param weight_dist: Type of distribution. Currently only supports ('normal', 'bernoulli')
        :param input_size: Input size of the layer
        :param output_size: Number of output neurons
        :param activation: Layer activation. Currently only supports ('relu', 'sigmoid', 'linear', 'softmax')
        :param use_bias: Whether to use bias nodes or not
        :param kwargs: Should be dict of dict containing 'weight_kwargs', 'bias_kwargs'
        """
        super().__init__()
        self.name = name
        self.neurons = neurons
        self.input_size = input_size
        self.use_bias = use_bias
        self.n_groups = n_groups
        self.mu = mu
        self.sd = sd

        self.weights_grp = None
        self.weights_sd = None
        self.weights_raw = None
        self.weights = None
        self.bias = None

        if isinstance(activation, str):
            if activation == 'linear' or activation is None:
                self.activation = None
            else:
                self.activation = retrieve_activation(activation)

    def build(self, *args):
        self.input = concatenate(args, axis=2)

        shape = (self.input_size, self.neurons)
        raw_shape = (self.n_groups, self.input_size, self.neurons)

        self.weights_grp = pm.Normal('{}_weights_grp'.format(self.name), mu=self.mu, sd=self.sd, shape=shape)
        self.weights_sd = pm.HalfNormal('{}_sd'.format(self.name), sd=1.)
        self.weights_raw = pm.Normal('{}_weights_raw'.format(self.name), mu=0., sd=self.sd, shape=raw_shape)
        self.weights = self.weights_raw * self.weights_sd + self.weights_grp
        if self.use_bias:
            self.bias = pm.Normal('{}_bias'.format(self.name), mu=0., sd=1., shape=self.neurons)
        self.built = True

    def __call__(self, *args, **kwargs):
        if not self.built:
            self.build(*args)
        act = tt.batched_dot(self.input, self.weights)
        if self.use_bias:
            act = act + self.bias
        if self.activation is None:
            return act
        else:
            return self.activation(act)


class BayesianConv2D(Layer):
    def __init__(self, name, filters, channels, filter_size, activation=None, use_bias=True, mu=0., sd=.5):
        """

        :param name:
        :param input_shape: The input shape
        :param output_size: Number of filters to use
        :param filter_size: filter size (3, 3) is pretty standard
        :param activation:
        :param use_bias:
        :param sd:
        """
        super().__init__()
        self.name = name
        self.filters = filters
        self.channels = channels
        self.filter_size = filter_size
        self.mu = mu
        self.sd = sd
        self.use_bias = use_bias
        self.weights = None
        self.bias = None

        if isinstance(activation, str):
            if activation == 'linear' or activation is None:
                self.activation = None
            else:
                self.activation = retrieve_activation(activation)

    def build(self, *args):
        self.input = concatenate(args, axis=1)
        shape = (self.filters, self.channels, self.filter_size[0], self.filter_size[1])
        self.weights = pm.Normal('{}_weights'.format(self.name), mu=np.transpose(self.mu), sd=self.sd, shape=shape)
        if self.use_bias:
            self.bias = pm.Normal('{}_bias'.format(self.name), mu=0., sd=1., shape=self.filters)
        self.built = True

    def __call__(self, *args, **kwargs):
        if not self.built:
            self.build(*args)
        act = nn.conv2d(self.input, self.weights)
        if self.use_bias:
            act = act + self.bias.dimshuffle('x', 0, 'x', 'x')
        if self.activation is None:
            return act
        else:
            return self.activation(act)


class MaxPooling2D(Layer):
    def __init__(self, pooling_size: tuple):
        super().__init__()
        self.pooling_size = pooling_size

    def __call__(self, *args, **kwargs):
        input = concatenate(args, axis=1)
        return pool_2d(input, self.pooling_size)


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        input = concatenate(args, axis=1)
        return input.flatten(2)
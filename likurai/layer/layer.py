from .. import floatX

import pymc3 as pm
from theano.tensor import concatenate
import theano.tensor.nnet as nn


def retrieve_distribution(dist):
    return getattr(pm, dist)


def retrieve_activation(act):
    return getattr(nn, act)


class Layer:
    def __init__(self):
        # TODO: Implement something like the Keras functional API
        # self.inputs = []
        pass

    def __call__(self, *args, **kwargs):
        pass


class BayesianDenseLayer(Layer):
    def __init__(self, name, input_size=None, output_size=None, shape=None, activation: str or function=None, sd=0.5,
                 use_bias=True, **kwargs):
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

        if shape is None and (input_size is None or output_size is None):
            raise ValueError("Must either provide shape or set input and output size")
        if shape is None:
            shape = (input_size, output_size)

        self.use_bias = use_bias
        self.weights = pm.Normal('{}_weights'.format(name), mu=0., sd=sd, shape=shape)
        if self.use_bias:
            self.bias = pm.Normal('{}_bias'.format(name), mu=0., sd=1., shape=shape[1])

        if isinstance(activation, str):
            if activation == 'linear' or activation is None:
                self.activation = None
            else:
                self.activation = retrieve_activation(activation)

    def __call__(self, *args, **kwargs):
        input = concatenate(args, axis=1)
        act = pm.math.dot(input, self.weights)
        if self.use_bias:
            act = act + self.bias
        if self.activation is None:
            return act
        else:
            return self.activation(act)


class HierarchicalBayesianDenseLayer(Layer):
    def __init__(self, name, input_size=None, output_size=None, n_groups=None, shape=None, sd=.5,
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

        if shape is None and (input_size is None or output_size is None):
            raise ValueError("Must either provide shape or set input and output size")
        if shape is None:
            shape = (input_size, output_size)

        self.use_bias = use_bias
        self.n_groups = n_groups

        self.weights = pm.Normal('{}_weights'.format(name), mu=0., sd=sd, shape=shape)
        self.weights_sd = pp
        if self.use_bias:
            self.bias = pm.Normal('{}_bias'.format(name), mu=0., sd=1., shape=shape[1])

        weights_in_grp = pm.Normal('w_in', mu=0., sd=1., shape=(input_size, output_size), testval=init_in)
        weights_in_sd = pm.HalfNormal('w_in_sd', sd=1.)
        weights_in_raw = pm.Normal('w_raw_in', mu=0., sd=1., shape=(self.n_groups, n_features, self.hidden_size))
        weights_in = weights_in_raw * weights_in_sd + weights_in_grp

        self.weights_grp = self.weight_dist('{}_weights_grp'.format(name), **kwargs['weight_kwargs'])


        if isinstance(activation, str):
            if activation == 'linear' or activation is None:
                self.activation = None
            else:
                self.activation = retrieve_activation(activation)

    def __call__(self, *args, **kwargs):
        input = concatenate(args, axis=1)
        act = pm.math.dot(input, self.weights)
        if self.use_bias:
            act = act + self.bias
        if self.activation is None:
            return act
        else:
            return self.activation(act)


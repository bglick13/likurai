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
    def __init__(self, name, weight_dist, input_size=None, output_size=None, activation: str or function = None,
                 use_bias=True, bias_dist=None, **kwargs):
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

        if 'shape' not in kwargs['weight_kwargs'] and (input_size is None or output_size is None):
            raise ValueError("Must either provide shape as a kwarg or set input and output size")
        if 'shape' not in kwargs and input_size is not None and output_size is not None:
            kwargs['weight_kwargs']['shape'] = (input_size, output_size)
            kwargs['bias_kwargs']['shape'] = output_size

        self.use_bias = use_bias
        self.weight_dist = retrieve_distribution(weight_dist)
        if bias_dist is None:
            self.bias_dist = retrieve_distribution(weight_dist)
        else:
            self.bias_dist = retrieve_distribution(bias_dist)
        self.weights = self.weight_dist('{}_weights'.format(name), **kwargs['weight_kwargs'])
        if self.use_bias:
            self.bias = self.bias_dist('{}_bias'.format(name), **kwargs['bias_kwargs'])

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
    def __init__(self, name, weight_dist, input_size=None, output_size=None, n_groups=None, activation: str or function = None,
                 use_bias=True, bias_dist=None, **kwargs):
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

        if 'shape' not in kwargs['weight_kwargs'] and (input_size is None or output_size is None):
            raise ValueError("Must either provide shape as a kwarg or set input and output size")
        if 'shape' not in kwargs and input_size is not None and output_size is not None:
            kwargs['weight_kwargs']['shape'] = (input_size, output_size)
            kwargs['bias_kwargs']['shape'] = output_size

        self.use_bias = use_bias
        self.n_groups = n_groups

        self.weight_dist = retrieve_distribution(weight_dist)
        if bias_dist is None:
            self.bias_dist = retrieve_distribution(weight_dist)
        else:
            self.bias_dist = retrieve_distribution(bias_dist)

        weights_in_grp = pm.Normal('w_in', mu=0., sd=1., shape=(n_features, self.hidden_size), testval=init_in)
        weights_in_sd = pm.HalfNormal('w_in_sd', sd=1.)
        weights_in_raw = pm.Normal('w_raw_in', mu=0., sd=1., shape=(self.n_groups, n_features, self.hidden_size))
        weights_in = weights_in_raw * weights_in_sd + weights_in_grp

        self.weights = self.weight_dist('{}_weights'.format(name), **kwargs['weight_kwargs'])
        if self.use_bias:
            self.bias = self.bias_dist('{}_bias'.format(name), **kwargs['bias_kwargs'])

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


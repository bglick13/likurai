from .. import floatX

import pymc3 as pm
from theano.tensor import concatenate
import theano.tensor.nnet as nn
import theano.tensor as tt


def retrieve_distribution(dist):
    return getattr(pm, dist)


def retrieve_activation(act):
    return getattr(nn, act)


class Layer:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
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
        self.dist = getattr(pm, dist)
        self.connected_param = connected_param

    def __call__(self, *args, **kwargs):
        input = concatenate(args, axis=1)
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


class HierarchicalBayesianDense(Layer):
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
            grp_shape = (n_groups, input_size, output_size)

        self.use_bias = use_bias
        self.n_groups = n_groups

        self.weights_grp = pm.Normal('{}_weights_grp'.format(name), mu=0., sd=sd, shape=shape)
        self.weights_sd = pm.HalfNormal('{}_sd', sd=1.)
        self.weights_raw = pm.Normal('{}_weights_raw', mu=0., sd=sd, shape=grp_shape)
        self.weights = self.weights_grp + self.weights_raw * self.weights_sd
        if self.use_bias:
            self.bias = pm.Normal('{}_bias'.format(name), mu=0., sd=1., shape=shape[1])

        if isinstance(activation, str):
            if activation == 'linear' or activation is None:
                self.activation = None
            else:
                self.activation = retrieve_activation(activation)

    def __call__(self, *args, **kwargs):
        input = concatenate(args, axis=1)
        act = tt.batched_dot(input, self.weights)
        if self.use_bias:
            act = act + self.bias
        if self.activation is None:
            return act
        else:
            return self.activation(act)


class BayesianConv2D(Layer):
    def __init__(self, name, input_shape, output_size, filter_size, activation=None, use_bias=True, sd=.5):
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
        self.use_bias = use_bias
        self.weight_shape = (filter_size[0], filter_size[1], input_shape[-1], output_size)
        self.weights = pm.Normal('{}_weights'.format(name), mu=0., sd=sd, shape=self.weight_shape)

        if self.use_bias:
            self.bias = pm.Normal('{}_bias'.format(name), mu=0., sd=1., shape=output_size)

        if isinstance(activation, str):
            if activation == 'linear' or activation is None:
                self.activation = None
            else:
                self.activation = retrieve_activation(activation)

    def __call__(self, *args, **kwargs):
        input = concatenate(args, axis=1)
        act = nn.conv2d(input, self.weights)
        if self.use_bias:
            act = act + self.bias
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
        return tt.signal.pool.pool_2d(input, self.pooling_size)


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        input = concatenate(args, axis=1)
        return input.flatten()
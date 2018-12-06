from .. import floatX

from . import pm
from . import retrieve_distribution, retrieve_activation
from . import concatenate


class Layer:
    def __init__(self):
        # TODO: Implement something like the Keras functional API
        # self.inputs = []
        pass

    def __call__(self, *args, **kwargs):
        pass


class BayesianDenseLayer(Layer):
    def __init__(self, name, type, input_size=None, output_size=None, activation: str or function = None, use_bias=True,
                 **kwargs):
        """
        Initialize a basic dense layer in a Bayesian framework
        :param name: Name of the layer
        :param type: Type of distribution. Currently only supports ('normal', 'bernoulli')
        :param input_size: Input size of the layer
        :param output_size: Number of output neurons
        :param activation: Layer activation. Currently only supports ('relu', 'sigmoid', 'linear', 'softmax')
        :param use_bias: Whether to use bias nodes or not
        :param kwargs: Should be dict of dict containing 'weight_kwargs', 'bias_kwargs'
        """
        super().__init__()

        if 'shape' not in kwargs['weight_kwargs'] and input_size is None or output_size is None:
            raise ValueError("Must either provide shape as a kwarg or set input and output size")
        if 'shape' not in kwargs and input_size is not None and output_size is not None:
            kwargs['weight_kwargs']['shape'] = (input_size, output_size)
            kwargs['bias_kwargs']['shape'] = output_size

        self.use_bias = use_bias
        self.dist = retrieve_distribution(type)
        self.weights = self.dist('{}_weights'.format(name), **kwargs['weight_kwargs'])
        self.bias = self.dist('{}_bias'.format(name), **kwargs['bias_kwargs'])

        if isinstance(activation, str):
            self.activation = retrieve_activation(activation)

    def __call__(self, *args, **kwargs):
        input = concatenate(args, axis=1)
        if self.activation is None:
            act = pm.math.dot(input, self.weights)
        else:
            act = self.activation(pm.math.dot(input, self.weights))
        if self.use_bias:
            act = act + self.bias
        return act





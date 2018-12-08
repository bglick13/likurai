"""
Defines the interface for a Model. Ideally, everything in this API will follow this signature, which will be similar
to scikit-learn's
"""
import pickle
import numpy as np
import pymc3 as pm
from theano import shared
from ..layer import Layer
import theano
floatX = theano.config.floatX


class Model:
    def __init__(self):
        # Model inputs/targets
        self.x = shared(np.zeros((1, 1)).astype(floatX))
        self.y = shared(np.array([]).astype(floatX))

        # Other model variables
        self.layers = []
        self.activations = []
        self.model = pm.Model()
        self.trace = None
        self.approx = None
        self.compiled = False

    def add_layer(self, layer: Layer):
        """
        Helper function to build simple models. Makes the assumption that the input to a layer is only the output of
        the previous layer.
        The output layer requires the following additional kwargs...
        - sd (model variance)
        - observed (shared variable)
        - total_size (necessary for minibatch training)
        - mu = model.activations[-1]
        :param layer:
        :return:
        """
        with self.model:
            if len(self.layers) == 0:
                self.activations.append(layer(self.x))
            else:
                self.activations.append(layer(self.activations[-1]))
            self.layers.append(layer)

    def compile(self, dist: str, connected_param, **kwargs):
        """
        Build the likelihood distribution the model will be conditioned on
        :param dist: the type of distribution to use
        :param connected_param: Which of the model params will be connected to the output layer
        :param kwargs: dict of dict describing the rest of the distribution parameters

        e.g.,
        ```
        bnn.compile('Normal', 'mu', {'sd': {'dist': 'HalfCauchy', 'name': 'sigma', 'beta': 5.}})
        ```
        :return:
        """
        dist = getattr(pm, dist)

        with self.model:
            _params = {connected_param: self.activations[-1]}
            jitter = kwargs.pop('jitter')
            _params[connected_param] += jitter
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    _dist = getattr(pm, value.pop('dist'))
                    _params[key] = _dist(**value)
                elif isinstance(value, [float, int]):
                    _params[key] = value

            likelihood = dist('likelihood', observed=self.y, total_size=len(self.x.get_value()), **_params)
        self.compiled = True

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


def load_model(filepath):
    model = pickle.load(open(filepath, 'rb'))
    return model


def save_model(model, filepath):
    pickle.dump(model, open(filepath, 'wb'))

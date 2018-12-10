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

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


def load_model(filepath):
    model = pickle.load(open(filepath, 'rb'))
    return model


def save_model(model, filepath):
    pickle.dump(model, open(filepath, 'wb'))

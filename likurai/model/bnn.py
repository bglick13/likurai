"""
Densely connected Bayesian Neural Network
"""

import numpy as np
import pymc3 as pm
from . import Model
from .. import shared
import pickle
from .. import floatX


class BayesianNeuralNetwork(Model):
    def __init__(self):
        super().__init__()
        # Model inputs/targets
        self.train_x = None  # Useful for introspection later
        self.train_y = None  # Useful for introspection later
        self.x = shared(np.zeros((1, 1)).astype(floatX))
        self.y = shared(np.array([]).astype(floatX))

        # Other model variables
        self.model = pm.Model()
        self.trace = []
        self.approx = []

    def fit(self, x, y, epochs=30000, method='advi', batch_size=128, n_models=1, **sample_kwargs):
        """

        :param x:
        :param y:
        :param epochs:
        :param method:
        :param batch_size: int or array. For hierarchical models, batch along the second dimension (e.g., [None, 128])
        :param sample_kwargs:
        :return:
        """
        self.train_x = x
        with self.model:
            if method == 'nuts':
                # self.x.set_value(x)
                # self.y.set_value(y)
                for _ in range(n_models):
                    self.trace.append(pm.sample(epochs, **sample_kwargs))
            else:
                mini_x = pm.Minibatch(x, batch_size=batch_size, dtype=floatX)
                mini_y = pm.Minibatch(y, batch_size=batch_size, dtype=floatX)

                if method == 'advi':
                    inference = pm.ADVI()
                elif method == 'svgd':
                    inference = pm.SVGD()
                for _ in range(n_models):
                    approx = pm.fit(n=epochs, method=inference, more_replacements={self.x: mini_x, self.y: mini_y}, **sample_kwargs)
                    self.trace.append(approx.sample(draws=20000))
                    self.approx.append(approx)

    def predict(self, x, n_samples=1, progressbar=True, point_estimate=False):
        self.x.set_value(x.astype(floatX))
        try:
            # For classification tasks
            self.y.set_value(np.zeros((np.array(x).shape[0], self.y.get_value().shape[1])).astype(floatX))
        except IndexError:
            # For regression tasks
            self.y.set_value(np.zeros((np.array(x).shape[0], 1)).astype(floatX))

        with self.model:
            ppc = None
            for trace in self.trace:
                _ppc = pm.sample_ppc(trace, samples=n_samples, progressbar=progressbar)['likelihood']
                if ppc is None:
                    ppc = _ppc
                else:
                    ppc = np.vstack((ppc, _ppc))
        if point_estimate:
            return np.mean(ppc, axis=0)
        return ppc

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump([self.model, self.trace, self.x, self.y, self.train_x, self.train_y], f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model, self.trace, self.x, self.y, self.train_x, self.train_y = pickle.load(f)

"""
Densely connected Bayesian Neural Network
"""

import numpy as np
import pymc3 as pm
from . import Model
import theano
floatX = theano.config.floatX


class BayesianNeuralNetwork(Model):
    def __init__(self):
        super().__init__()

    def fit(self, x, y, epochs=30000, method='advi', batch_size=128, **sample_kwargs):
        with self.model:
            if method == 'nuts':
                # self.x.set_value(x)
                # self.y.set_value(y)
                self.trace = pm.sample(epochs, init='advi', **sample_kwargs)
            else:
                mini_x = pm.Minibatch(x, batch_size=batch_size, dtype=floatX)
                mini_y = pm.Minibatch(y, batch_size=batch_size, dtype=floatX)

                if method == 'advi':
                    inference = pm.ADVI()
                elif method == 'svgd':
                    inference = pm.SVGD()
                approx = pm.fit(n=epochs, method=inference, more_replacements={self.x: mini_x,
                                                                               self.y: mini_y}, **sample_kwargs)
                self.trace = approx.sample(draws=20000)
                self.approx = approx

    def predict(self, x, n_samples=1, progressbar=True, point_estimate=False):
        self.x.set_value(x.astype(floatX))
        try:
            # For classification tasks
            self.y.set_value(np.zeros((np.array(x).shape[0], self.y.get_value().shape[1])).astype(floatX))
        except IndexError:
            # For regression tasks
            self.y.set_value(np.zeros((np.array(x).shape[0], 1)).astype(floatX))

        with self.model:
            ppc = pm.sample_ppc(self.trace, samples=n_samples, progressbar=progressbar)
        if point_estimate:
            return np.mean(ppc['likelihood'], axis=0)
        return ppc['likelihood']


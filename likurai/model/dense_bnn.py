"""
Densely connected Bayesian Neural Network
"""
from .. import floatX

from . import np
from . import pm
from . import Model
from . import tt
from . import BayesianDenseLayer


# def build_model(n_features, n_hidden, hidden_size, n_targets):
#     model = BayesianDenseNeuralNetwork()
#     for i in range(n_hidden):
#         input_size = hidden_size * i + n_features
#         model.add_layer(BayesianDenseLayer(i, input_size, hidden_size, activation='relu'))
#
#     input_size = hidden_size * (n_hidden + 1)
#     model.add_layer(BayesianDenseLayer('out', input_size, n_targets, activation='linear'))
#     return model


class BayesianDenseNeuralNetwork(Model):
    def __init__(self):
        super().__init__()

    # def build_layer_input(self, index):
    #     """
    #     Dense network specific. Layer input is the concatenation of all previous outputs along with the original model
    #     input
    #     :param index:
    #     :return:
    #     """
    #     layer_input = tt.concatenate(self.activations[:index], axis=1)
    #     layer_input = tt.concatenate([layer_input, self.x], axis=1)
    #     return layer_input

    # def compile(self, sd, total_size: int):
    #     """
    #     Build the computation tree and create the likelihood distribution with observed variable
    #
    #     :param sd: A pymc3 distribution representing model variance
    #     :param total_size: The total number of observations. Important for minibatch training
    #     :return:
    #     """
    #     with self.model:
    #         # Build the activations
    #         for i, l in enumerate(self.layers):
    #             self.activations.append(l(self.build_layer_input(i)))
    #
    #         likelihood = pm.Normal('likelihood', mu=self.activations[-1], sd=sd, observed=self.y, total_size=total_size)
    #     self.compiled = True

    def train(self, x, y, epochs=30000, method='advi', batch_size=128):
        with self.model:
            if method == 'nuts':
                self.x.set_value(x)
                self.y.set_value(y)
                self.trace = pm.sample(epochs, init='advi', chains=1, njobs=1, tune=5000)
            else:
                mini_x = pm.Minibatch(x, batch_size=batch_size)
                mini_y = pm.Minibatch(y, batch_size=batch_size)

                if method == 'advi':
                    inference = pm.ADVI()
                elif method == 'svgd':
                    inference = pm.SVGD()
                approx = pm.fit(n=epochs, method=inference, more_replacements={self.x: mini_x,
                                                                               self.y: mini_y})
                self.trace = approx.sample(draws=20000)
                self.approx = approx

    def predict(self, x, n_samples=1, progressbar=True, deterministic=False):
        self.x.set_value(x)
        self.y.set_value(np.zeros(np.array(x).shape[0]))
        with self.model:
            ppc = pm.sample_posterior_predictive(self.trace, samples=n_samples, progressbar=progressbar)
        if deterministic:
            return ppc['likelihood'].mean(axis=0)
        return ppc['likelihood']


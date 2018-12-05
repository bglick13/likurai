"""
Densely connected Bayesian Neural Network
"""
from .. import floatX

from . import np
from . import pm
from . import Model
from . import tt
from . import shared
from . import BayesianDenseLayer


class BayesianDenseNeuralNetwork(Model):
    def __init__(self, n_hidden: int, hidden_size: int or list):
        super().__init__()

        # Model size variables
        self.n_hidden = int(n_hidden)
        self.hidden_size = int(hidden_size)

        # Model inputs/targets
        self.x = shared(np.zeros((1, 1)))
        self.y = shared(np.array([]))

        # Other model variables
        self.layers = []
        self.activations = []
        self.model = pm.Model()
        self.trace = None
        self.approx = None
        self.compiled = False

    def add_layer(self, layer):
        with self.model:
            self.layers.append(layer)

    def build_layer_input(self, index):
        layer_input = tt.concatenate(self.activations[:index], axis=1)
        layer_input = tt.concatenate([layer_input, self.x], axis=1)
        return layer_input

    def compile(self, sd, total_size):
        with self.model:
            # Build the activations
            for i in range(self.n_hidden + 1):
                self.activations.append(self.layers[i](self.build_layer_input(i)))

            likelihood = pm.Normal('likelihood', mu=self.activations[-1], sd=sd, observed=self.y, total_size=total_size)
        self.compiled = True

    def build_model(self, n_features, n_targets, total_size):
        with self.model:
            for i in range(self.n_hidden - 1):
                input_size = self.hidden_size * i + n_features
                self.add_layer(BayesianDenseLayer(i, input_size, self.hidden_size, activation='relu'))

            input_size = self.hidden_size * (self.n_hidden + 1)
            self.add_layer(BayesianDenseLayer('out', input_size, n_targets, activation='linear'))

        sigma = pm.HalfCauchy('sigma', 5)  # Model variance
        self.compile(sigma, total_size)

    def train(self, x, y, epochs=30000, method='advi', batch_size=128):
        with self.model:
            if method == 'nuts':
                self.x.set_value(x)
                self.y.set_value(y)
                self.trace = pm.sample(epochs, init='advi', chains=1, njobs=1, tune=5000)
            else:
                mini_x = pm.Minibatch(x, batch_size=batch_size)
                mini_y = pm.Minibatch(y, batch_size=batch_size)

                inference = pm.ADVI()
                approx = pm.fit(n=epochs, method=inference, more_replacements={self.x: mini_x,
                                                                               self.y: mini_y})
                self.trace = approx.sample(draws=20000)
                self.approx = approx

    def predict(self, x, groups=None, n_samples=1, progressbar=True):
        self.x.set_value(x)
        self.y.set_value(np.zeros(np.array(x).shape[0]))
        with self.model:
            ppc = pm.sample_posterior_predictive(self.trace, samples=n_samples, progressbar=progressbar)
        return ppc['likelihood']

    def load_trace(self, filepath):
        self.trace = pickle.load(open(filepath, 'rb'))

    def save_trace(self, filepath):
        pickle.dump(self.trace, open(filepath, 'wb'))

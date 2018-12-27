"""
Densely connected Bayesian Neural Network
"""

import numpy as np
import pymc3 as pm
from . import Model
from .. import shared
import pickle
from .. import floatX
import tensorflow as tf
import tensorflow_probability as tfp


class TFPNetwork(Model):
    def __init__(self, x, y, n_hidden, hidden_size, output_size):
        super().__init__()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, x.shape[1]], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, y.shape[1]], name='y')
            self.batch_size = tf.placeholder(tf.int64, name='batch_size')

            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).shuffle(10000).batch(self.batch_size).repeat()
            self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(self.batch_size).repeat()

            self.iter = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)
            self.features, self.labels = self.iter.get_next()

            self.train_init_op = self.iter.make_initializer(self.train_dataset, name='train_dataset_init')
            self.test_init_op = self.iter.make_initializer(self.test_dataset, name='test_dataset_init')

            model = tf.keras.Sequential()
            for i in range(n_hidden):
                model.add(tfp.layers.DenseFlipout(hidden_size, activation=tf.nn.relu, name='layer_{}'.format(i)))
            model.add(tfp.layers.DenseFlipout(output_size, name='output'))
            self.model = model
            self.logits = self.model(self.features)
            self.labels_distribution = tfp.distributions.Normal(loc=self.logits, scale=1.0)

            neg_log_likelihood = -tf.reduce_mean(self.labels_distribution.log_prob(self.labels))
            kl = sum(model.losses) / len(y)
            self.elbo_loss = neg_log_likelihood + kl
            self.predictions = tf.identity(self.logits, name='prediction')
            self.accuracy, self.accuracy_update_op = tf.metrics.mean_absolute_error(labels=self.labels,
                                                                                    predictions=self.predictions)

            optimizer = tf.train.AdamOptimizer(0.001)
            self.train_op = optimizer.minimize(self.elbo_loss)

            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def fit(self, x, y, epochs, batch_size):
        n_batches = (len(x) // batch_size) + 1

        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init_op)
            for i in range(epochs):
                total_loss = 0
                sess.run(self.train_init_op, feed_dict={self.x: x, self.y: y, self.batch_size: batch_size})
                for _ in range(n_batches):
                    _, _, loss_value = sess.run([self.train_op, self.accuracy_update_op, self.elbo_loss])
                    total_loss += loss_value

                _, accuracy_value = sess.run([self.accuracy_update_op, self.accuracy])
                print("Iter: {}, Loss: {:.4f}, MAE: {:.4f}".format(i, total_loss / n_batches, accuracy_value))

    def predict(self, X, batch_size=None, n_draws=1):
        if batch_size is None:
            batch_size = len(X)

        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init_op)
            sess.run(self.test_init_op, feed_dict={self.x: X, self.y: np.ones((len(X), 1)), self.batch_size: batch_size})
            pred = np.asarray([sess.run(self.predictions) for _ in range(n_draws)]).mean(axis=0)

        return pred

    def save_model(self, filepath):
        with tf.Session(graph=self.graph):
            self.model.save_weights('{}'.format(filepath))

    def load_model(self, filepath):
        with tf.Session(graph=self.graph):
            self.model.load_weights('{}'.format(filepath))

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
        self.inference = []
        self.trace = []

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
        self.train_y = y
        with self.model:
            if method == 'nuts':
                # self.x.set_value(x)
                # self.y.set_value(y)
                for _ in range(n_models):
                    self.trace.append(pm.sample(epochs, **sample_kwargs))
            else:
                mini_x = pm.Minibatch(x, batch_size=batch_size, dtype=floatX)
                mini_y = pm.Minibatch(y, batch_size=batch_size, dtype=floatX)

                if len(self.inference) == 0:
                    for _ in range(n_models):
                        if method == 'advi':
                            self.inference.append(pm.ADVI())
                        elif method == 'svgd':
                            self.inference.append(pm.SVGD(n_particles=100))

                        approx = self.inference[_].fit(epochs, more_replacements={self.x: mini_x, self.y: mini_y}, **sample_kwargs)
                        # approx = pm.fit(n=epochs, method=inference, more_replacements={self.x: mini_x, self.y: mini_y}, **sample_kwargs)
                        self.trace.append(approx.sample(draws=10000))
                else:
                    print("Pre-trained model - refining fit")
                    for i, inf in enumerate(self.inference):
                        inf.refine(epochs)
                        self.trace[i] = inf.approx.sample(draws=10000)

    def predict(self, x, n_samples=1, progressbar=True, point_estimate=False, **kwargs):
        self.x.set_value(x.astype(floatX))
        if len(x.shape) == 3:
            # Hierarchical model
            self.y.set_value(np.zeros((np.array(x).shape[0], x.shape[1], self.y.get_value().shape[2])).astype(floatX))
        else:
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
            pickle.dump([self.model, self.inference, self.trace, self.x, self.y, self.train_x, self.train_y], f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model, self.inference, self.trace, self.x, self.y, self.train_x, self.train_y = pickle.load(f)


class HierarchicalBayesianNeuralNetwork(BayesianNeuralNetwork):
    def __init__(self):
        super().__init__()

    def prepare_train_data(self, x, y, groups):
        Xs, Ys = [], []
        min_len = np.inf
        for i in np.unique(groups):
            X = np.array(x[groups == i]).astype(float)
            Y = np.array(y[groups == i]).astype(float)
            if len(X) < min_len:
                min_len = len(X)
            Xs.append(X)
            Ys.append(Y)

        Xss, Yss = [], []
        for _x, _y in zip(Xs, Ys):
            _x = _x[:min_len, :]
            _y = _y[:min_len]

            Xss.append(_x)
            Yss.append(_y)

        x = np.stack(Xss)
        y = np.stack(Yss)
        return x, y

    def prepare_test_data(self, x, groups):
        Xs = []
        max_len = 0
        for i in np.unique(groups):
            X = np.array(x[groups == i]).astype(float)
            if len(X) > max_len:
                max_len = len(X)
            Xs.append(X)

        Xss = []
        mask_idxs = []
        for _x in Xs:
            app_x = np.ones((max_len - len(_x), _x.shape[1]))  # * np.nan
            mask_idxs.append(len(_x))
            _x = np.vstack((_x, app_x))
            Xss.append(_x)

        x = np.stack(Xss)
        return x

    def fit(self, x, y, epochs=30000, method='advi', batch_size=128, n_models=1, **sample_kwargs):
        """
        X, y should already be formatted into extra dimension for groups
        :param x:
        :param y:
        :param epochs:
        :param method:
        :param batch_size:
        :param n_models:
        :param sample_kwargs:
        :return:
        """
        super().fit(x, y, epochs, method, batch_size, n_models, **sample_kwargs)

    def predict(self, x, n_samples=1, progressbar=True, point_estimate=False, **kwargs):
        groups = kwargs.pop('groups')
        ppc = super().predict(x, n_samples, progressbar, point_estimate, **kwargs)
        grp_count = dict((g, 0) for g in np.unique(groups))
        grp_occurrences = []
        for g in groups:
            grp_occurrences.append(grp_count[g])
            grp_count[g] += 1

        print(grp_occurrences)
        pred = ppc[:, groups, grp_occurrences]
        return pred
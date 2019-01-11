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
from tensorflow_probability.python.layers.util import default_mean_field_normal_fn
from tensorflow.python.saved_model import tag_constants
from functools import partial
import shutil


class TFPNetwork(Model):
    def __init__(self, filepath):
        super().__init__()

        self.filepath = filepath

    def build_model(self, x, y, n_hidden, hidden_size, learning_rate, overwrite=True):
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                x_ph = tf.placeholder(tf.float32, shape=[None, x.shape[1]], name='x')
                y_ph = tf.placeholder(tf.float32, shape=[None], name='y')
                batch_size = tf.placeholder(tf.int64, name='batch_size')

                train_dataset = tf.data.Dataset.from_tensor_slices((x_ph, y_ph)).shuffle(10000).batch(
                    batch_size).repeat()
                test_dataset = tf.data.Dataset.from_tensor_slices((x_ph, y_ph)).batch(batch_size).repeat()

                iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
                features, labels = iter.get_next()

                train_init_op = iter.make_initializer(train_dataset, name='train_dataset_init')
                test_init_op = iter.make_initializer(test_dataset, name='test_dataset_init')

                input = tf.keras.Input(shape=(x.shape[1],))
                h = input
                layers = [h]
                for i in range(n_hidden):
                    h = tfp.layers.DenseFlipout(hidden_size // np.power(2, i), activation=tf.nn.relu,
                                                name='layer_{}'.format(i),
                                                kernel_posterior_fn=default_mean_field_normal_fn(
                                                    loc_initializer=tf.random_normal_initializer(stddev=1.0),
                                                ))(h)
                    layers.append(h)
                    h = tf.keras.layers.Concatenate()(layers)

                loc_output = tfp.layers.DenseFlipout(1, name='loc_output',
                                                     kernel_posterior_fn=default_mean_field_normal_fn(
                                                         loc_initializer=tf.random_normal_initializer(stddev=1.0),
                                                     ))(h)
                model = tf.keras.Model(inputs=input, outputs=loc_output)
                model.summary()

                _loc = model(features)
                # _loc = model(features)
                labels_distribution = tfp.distributions.Normal(loc=_loc, scale=0.5)

                label_probs = labels_distribution.sample(name='label_probs')

                neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
                kl = sum(model.losses) / len(y)
                elbo_loss = neg_log_likelihood + kl
                elbo_loss = tf.identity(elbo_loss, name='elbo_loss')
                # predictions = tf.argmax(_loc, axis=1, name='prediction')
                predictions = tf.reduce_mean(_loc, axis=1, name='prediction')
                # accuracy, accuracy_update_op = tf.metrics.accuracy(labels=labels,
                #                                                    predictions=predictions,
                #                                                    name='accuracy')
                accuracy, accuracy_update_op = tf.metrics.mean_absolute_error(labels=labels,
                                                                              predictions=predictions,
                                                                              name='accuracy')

                optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(elbo_loss, name='train_op')

                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name='init_op')
                sess.run(init_op)
                inputs = {
                    "batch_size": batch_size,
                    "features": features,
                    "labels": labels,
                }
                outputs = {"prediction": predictions}
                if overwrite:
                    shutil.rmtree('{}/{}'.format(self.filepath, 'build'), ignore_errors=True)

                tf.saved_model.simple_save(
                    sess, '{}/{}'.format(self.filepath, 'build'), inputs, outputs
                )

    def fit(self, x, y, epochs, batch_size, val_x=None, val_y=None, val_batch=None, overwrite=True):
        n_batches = (len(x) // batch_size) + 1
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                try:
                    tf.saved_model.loader.load(
                        sess,
                        [tag_constants.SERVING],
                        '{}/{}'.format(self.filepath, 'fit')
                    )
                    print("Successfully loaded checkpoint")
                except:
                    tf.saved_model.loader.load(
                        sess,
                        [tag_constants.SERVING],
                        '{}/{}'.format(self.filepath, 'build')
                    )
                # Get restored placeholders
                labels_data_ph = graph.get_tensor_by_name('y:0')
                features_data_ph = graph.get_tensor_by_name('x:0')
                batch_size_ph = graph.get_tensor_by_name('batch_size:0')
                # Get restored model output
                restored_logits = graph.get_tensor_by_name('prediction:0')
                train_op = graph.get_operation_by_name('train_op')
                accuracy_update_op = graph.get_tensor_by_name('accuracy/update_op:0')
                accuracy = graph.get_tensor_by_name('accuracy/value:0')
                elbo_loss = graph.get_tensor_by_name('elbo_loss:0')
                # Get dataset initializing operation
                dataset_init_op = graph.get_operation_by_name('train_dataset_init')
                test_init_op = graph.get_operation_by_name('test_dataset_init')
                init_op = graph.get_operation_by_name('init_op')
                sess.run(init_op)
                sess.run(dataset_init_op, feed_dict={features_data_ph: x,
                                                     labels_data_ph: y,
                                                     batch_size_ph: batch_size})

                for i in range(epochs):
                    if val_x is not None:
                        sess.run(dataset_init_op, feed_dict={features_data_ph: x,
                                                             labels_data_ph: y,
                                                             batch_size_ph: batch_size})
                    total_loss = 0
                    for _ in range(n_batches):
                        _, _, loss_value = sess.run([train_op, accuracy_update_op, elbo_loss])
                        total_loss += loss_value
                    if i % 10 == 0:
                        print("Predicted means/variance...")
                        means = sess.run(restored_logits)
                        print(np.round(means, 3))
                        print(np.std(means))

                    _, accuracy_value = sess.run([accuracy_update_op, accuracy])
                    print("Iter: {}, Loss: {:.4f}, MAE: {:.4f}".format(i, total_loss / n_batches, accuracy_value))
                    if val_x is not None:
                        sess.run(test_init_op, feed_dict={features_data_ph: val_x,
                                                          labels_data_ph: val_y,
                                                          batch_size_ph: val_batch})
                        _, val_accuracy = sess.run([accuracy_update_op, accuracy])
                        print("VAL MAE: {:.4f}".format(val_accuracy))

                inputs = {
                    "batch_size": batch_size_ph,
                    "features": features_data_ph,
                    "labels": labels_data_ph,
                }
                outputs = {"prediction": restored_logits}
                if overwrite:
                    shutil.rmtree('{}/{}'.format(self.filepath, 'fit'), ignore_errors=True)
                tf.saved_model.simple_save(
                    sess, '{}/{}'.format(self.filepath, 'fit'), inputs, outputs
                )

    def predict(self, X, y=None, batch_size=None, n_draws=1, sample=False):
        if batch_size is None:
            batch_size = len(X)
        if y is None:
            y = np.ones(len(X))

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                tf.saved_model.loader.load(
                    sess,
                    [tag_constants.SERVING],
                    '{}/{}'.format(self.filepath, 'fit')
                )
                # Get restored placeholders
                labels_data_ph = graph.get_tensor_by_name('y:0')
                features_data_ph = graph.get_tensor_by_name('x:0')
                batch_size_ph = graph.get_tensor_by_name('batch_size:0')
                # Get restored model output
                restored_logits = graph.get_tensor_by_name('prediction:0')
                label_probs = graph.get_tensor_by_name('Normal/label_probs/Reshape:0')
                # Get dataset initializing operation
                dataset_init_op = graph.get_operation_by_name('test_dataset_init')

                # Initialize restored dataset
                sess.run(
                    dataset_init_op,
                    feed_dict={
                        features_data_ph: X,
                        labels_data_ph: y,
                        batch_size_ph: batch_size
                    })
                if sample:
                    pred = np.asarray([sess.run(label_probs) for _ in range(n_draws)])
                else:
                    pred = np.asarray([sess.run(restored_logits) for _ in range(n_draws)])

        return pred


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
        self.inference = None
        self.trace = None
        self.total_train_runs = 0

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
                self.x.set_value(x.astype(floatX))
                self.y.set_value(y.astype(floatX))
                for _ in range(n_models):
                    if self.trace is not None:
                        trace = self.trace
                    else:
                        trace = None
                    self.trace = pm.sample(epochs, trace=trace, **sample_kwargs)
            else:
                mini_x = pm.Minibatch(x, batch_size=batch_size, dtype=floatX)
                mini_y = pm.Minibatch(y, batch_size=batch_size, dtype=floatX)

                if self.inference is None:
                    if method == 'advi':
                        self.inference = pm.ADVI()
                    elif method == 'svgd':
                        self.inference = pm.SVGD(n_particles=100)

                    approx = self.inference.fit(epochs, more_replacements={self.x: mini_x, self.y: mini_y},
                                                **sample_kwargs)
                    self.trace = approx.sample(draws=10000)
                else:
                    print("Pre-trained model - refining fit")

                    self.inference.refine(epochs)
                    self.trace = self.inference.approx.sample(draws=10000)
        self.total_train_runs += epochs

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
            ppc = pm.sample_ppc(self.trace, samples=n_samples, progressbar=progressbar)['likelihood']

        if point_estimate:
            return np.mean(ppc, axis=0)
        return ppc

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump([self.model, self.total_train_runs, self.inference, self.trace, self.x, self.y, self.train_x, self.train_y], f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model, self.total_train_runs, self.inference, self.trace, self.x, self.y, self.train_x, self.train_y = pickle.load(f)


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
        original_indexes = np.array([i for i in range(len(x))])
        max_idx = len(x)
        Xs = []
        Idxs = []
        max_len = 0
        for i in np.unique(groups):
            X = np.array(x[groups == i]).astype(float)
            idx = np.array(original_indexes[groups == i])
            if len(X) > max_len:
                max_len = len(X)
            Xs.append(X)
            Idxs.append(idx)

        Xss = []
        mask_idxs = []
        for _x in Xs:
            app_x = np.ones((max_len - len(_x), _x.shape[1]))  # * np.nan
            mask_idxs.append(len(_x))
            _x = np.vstack((_x, app_x))
            Xss.append(_x)

        x = np.stack(Xss)
        return x, Idxs, max_idx

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
        idxs = kwargs.pop('idxs')
        max_idx = kwargs.pop('max_idx')
        ppc = super().predict(x, n_samples, progressbar, point_estimate, **kwargs)
        out = np.empty((n_samples, max_idx))

        # For each group
        for grp, idxs_of_group in enumerate(idxs):
            # i: the index in the group output, original_i: the original index in the flat input
            for i, original_i in enumerate(idxs_of_group):
                out[:, original_i] = ppc[:, grp, i].squeeze()
        return out
        # # Keeps track of how many times a sample from each group has appeared
        # grp_count = dict((g, 0) for g in np.unique(groups))
        # # Used to index the group occurrences into a flat array to return
        # grp_occurrences = []
        # for g in groups:
        #     grp_occurrences.append(grp_count[g])
        #     grp_count[g] += 1
        #
        # pred = ppc[:, groups, grp_occurrences]
        # return pred

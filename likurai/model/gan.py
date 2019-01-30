"""
This file will contain objects to create various forms of GANs
"""
import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Input, Concatenate, Dense, TimeDistributed, RepeatVector, Embedding, Multiply
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
import os
import tensorflow as tf
import tensorflow_probability as tfp
floatX = tf.float32


class GAN:

    def __init__(self):
        self.discriminator = None
        self.generator = None
        self.combined = None
        self.build_models()

    def load_model(self, filepath):
        self.discriminator = load_model(os.path.join(filepath, 'discriminator.h5'))
        self.generator = load_model(os.path.join(filepath, 'generator.h5'))
        self.combined = load_model(os.path.join(filepath, 'combined.h5'))

    def save_model(self, filepath):
        self.discriminator.save(os.path.join(filepath, 'discriminator.h5'))
        self.generator.save(os.path.join(filepath, 'generator.h5'))
        self.combined.save(os.path.join(filepath, 'combined.h5'))

    def build_discriminator(self):
        return Model()

    def build_generator(self):
        return Model()

    def build_models(self):
        pass

# class Generator:
#
#     def __init__(self):


class ConditionalSeqGAN:
    class Generator:
        def __init__(self, n_classes, conditional_data_size, embedding_dim, hidden_size, learning_rate, sequence_length,
                     batch_size):
            self.n_classes = n_classes
            self.conditional_data_size = conditional_data_size
            self.embedding_dim = embedding_dim
            self.hidden_size = hidden_size
            self.learning_rate = learning_rate
            self.sequence_length = sequence_length
            self.batch_size = batch_size

            # sequence_input =

            sequence_input = Input(batch_shape=(self.sequence_length, self.n_classes))
            conditional_input = Input((conditional_data_size,))
            embedding = Embedding(n_classes, embedding_dim, mask_zero=True)(sequence_input)
            conditional_embedding = Dense(embedding_dim, activation='tanh')(conditional_input)
            h = Multiply()([embedding, conditional_embedding])
            Model(inputs=[sequence_input, conditional_input], outputs=h).summary()
            # Stateful = True should allow MCTS style prediction (need to manually reset states each batch during train)
            self.lstm = LSTM(hidden_size, return_sequences=False, return_state=True, stateful=True, activation='tanh')
            self.states = self.lstm(h)
            self.lstm_output = self.states[0]
            self.states = self.states[1:]

            # Not time distributed since we're doing MCTS style prediction
            logits = Dense(n_classes, activation='softmax')(self.lstm_output)
            self.model = Model(inputs=[sequence_input, conditional_input], outputs=logits)
            self.model.summary()
            self.opt = Adam(learning_rate)

        def sample(self, num_samples, sequence_input, conditional_data):
            """
            Generate num_samples sequences given the input data
            :param num_samples:
            :param conditional_data:
            :return:
            """
            # inp = np.zeros((num_samples, self.sequence_length, self.n_classes))
            output = np.tile(sequence_input, (num_samples, 1, 1))
            for i in range(num_samples):
                for j in range(self.sequence_length):
                    if output[i, j].sum() > 0:
                        continue
                    out = self.model.predict_on_batch([output[i], conditional_data.reshape((1, -1))])
                    # self.lstm.reset_states(states=self.states)
                    choice = np.random.choice(range(self.n_classes), size=1, p=out[j])
                    output[i, j, choice] = 1
                self.model.reset_states()
            return output

        def pg_loss(self, sample, reward):
            """
            :param sample:
            :param target:
            :param reward:
            :return:
            """

            """
            self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
        )
            """
            loss = -K.sum(K.log(sample) * reward)

            return loss

        def train_pg(self, s, rewards):

            loss = self.pg_loss(s, rewards)  # Target should just remove the start letter/sequence
            updates = self.opt.get_updates(self.model.trainable_weights, [], loss)
            self.train_fn = K.function(inputs=[s, rewards],
                                       outputs=[],
                                       updates=updates)
            self.train_fn([s, rewards])
            return loss

    class Discriminator:

        def __init__(self, n_classes, conditional_data_size, embedding_dim, hidden_size, learning_rate,
                     sequence_length):
            self.n_classes = n_classes
            self.conditional_data_size = conditional_data_size
            self.embedding_dim = embedding_dim
            self.hidden_size = hidden_size
            self.learning_rate = learning_rate
            self.sequence_length = sequence_length

            sequence_input = Input(batch_shape=(self.sequence_length, self.n_classes))
            conditional_input = Input((self.conditional_data_size,))

            # Embed both inputs to the same space
            embedding = Embedding(self.n_classes, self.embedding_dim)(sequence_input)
            conditional_embedding = Dense(self.embedding_dim, activation='tanh')(conditional_input)
            h = Multiply()([embedding, conditional_embedding])
            Model(inputs=[sequence_input, conditional_input], outputs=h).summary()

            # LSTM layer - statefulness isn't necessary as w're predicting for each timestep in a sequence
            h = LSTM(self.hidden_size, return_sequences=True, activation='tanh')(h)
            logits = TimeDistributed(Dense(1, activation='sigmoid'))(h)
            self.model = Model(inputs=[sequence_input, conditional_input], outputs=logits)
            self.model.compile(Adam(self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
            self.model.summary()

        def fit(self, sequence_x, conditional_x, y, batch_size=64):
            self.model.fit([sequence_x, conditional_x], y, batch_size=batch_size)

        def train_on_batch(self, sequence_x, conditional_x, y):
            d_loss = self.model.train_on_batch([sequence_x, conditional_x], y)
            return d_loss

        def predict(self, sequence_x, conditional_x):
            return self.model.predict_on_batch([sequence_x, conditional_x.reshape((1, -1))])

    # TODO: Convert model to raw tensorflow (maybe - try to get working in keras first)

    def __init__(self, discriminator_hidden_size, embedding_dim, generator_hidden_size,
                 conditional_data_size, n_classes, sequence_length, learning_rate, batch_size=64):
        self.embedding_dim = embedding_dim
        self.conditional_data_size = conditional_data_size
        self.generator_hidden_size = generator_hidden_size
        self.discriminator_hidden_size = discriminator_hidden_size
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.generator = ConditionalSeqGAN.Generator(self.n_classes, self.conditional_data_size, self.embedding_dim,
                                                     self.generator_hidden_size, self.learning_rate,
                                                     self.sequence_length, batch_size)
        self.discriminator = ConditionalSeqGAN.Discriminator(self.n_classes, self.conditional_data_size,
                                                             self.embedding_dim, self.discriminator_hidden_size,
                                                             self.learning_rate, self.sequence_length)

    def train(self, sequence_data, conditional_data, epochs, batch_size=64):

        # 1) Pre-train generator
        pass

        # 2) Pre-train discriminator
        pass

        # 3) Adversarial training
        valid = np.ones((batch_size, self.sequence_length, 1))
        fake = np.zeros((batch_size, self.sequence_length, 1))
        start_sequence = np.zeros((18, self.n_classes))

        for epoch in range(epochs):
            idxs = np.random.randint(0, len(conditional_data), batch_size)
            generated_samples = []
            # Train generator
            g_loss = 0
            for idx in idxs:
                s = self.generator.sample(batch_size, start_sequence, conditional_data[idx])
                rewards = np.array([self.discriminator.predict(s[i], conditional_data[idx]) for i in range(batch_size)]).squeeze()
                g_loss += self.generator.train_pg(s, rewards)
                generated_samples.append(s)

            g_loss /= batch_size

            sequence_batch = sequence_data[idxs]
            conditional_batch = conditional_data[idxs]
            d_loss_real = self.discriminator.train_on_batch(sequence_batch, conditional_batch, valid)
            d_loss_fake = self.discriminator.train_on_batch(np.array(generated_samples), conditional_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))


# TODO: Implement MCTS (i.e., the prediction at a certain timestep should be dependent on its previous predictions)
# TODO: Implement SeqGAN (i.e., GAN that works on sequences of discrete data)
# TODO: Implement experience replay for generators (i.e., train on its previously generated samples)
class ConditionalRCGAN(GAN):

    def __init__(self, n_discriminator_layers, discriminator_hidden_size, n_generator_layers, generator_hidden_size,
                 generator_embed_size, generator_latent_dim, conditional_data_size, n_classes, sequence_length, learning_rate):
        self.n_discriminator_layers = n_discriminator_layers
        self.n_generator_layers = n_generator_layers
        self.conditional_data_size = conditional_data_size
        self.generator_hidden_size = generator_hidden_size
        self.generator_latent_dim = generator_latent_dim
        self.generator_embed_size = generator_embed_size
        self.discriminator_hidden_size = discriminator_hidden_size
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        super().__init__()

    def build_discriminator(self):
        inputs = Input(shape=(self.sequence_length, 1))
        c = Input(shape=(self.conditional_data_size, ))
        c_stacked = RepeatVector(self.sequence_length)(c)
        h = Concatenate(axis=2)([inputs, c_stacked])
        for _ in range(self.n_discriminator_layers):
            h = LSTM(self.discriminator_hidden_size, return_sequences=True, activation='tanh')(h)
        output = TimeDistributed(Dense(1, activation='sigmoid'))(h)
        model = Model(inputs=[inputs, c], outputs=output)
        print("Discriminator summary...")
        model.summary()
        return model

    def build_generator(self):
        z = Input(shape=(self.sequence_length, self.generator_latent_dim))
        c = Input(shape=(self.conditional_data_size, ))
        c_embed = Dense(self.generator_embed_size, activation='tanh')(c)
        c_stacked = RepeatVector(self.sequence_length)(c_embed)
        h = Concatenate(axis=2)([z, c_stacked])
        for _ in range(self.n_generator_layers):
            h = LSTM(self.generator_hidden_size, return_sequences=True, activation='tanh')(h)
        outputs = TimeDistributed(Dense(1, activation='tanh'))(h)
        model = Model(inputs=[z, c], outputs=outputs)
        print("Generator summary...")
        model.summary()
        return model

    def build_models(self):
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.discriminator.compile(Adam(self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        self.generator.compile(Adam(self.learning_rate), loss='mean_squared_error')

        z = Input(shape=(self.sequence_length, self.generator_latent_dim))
        c = Input(shape=(self.conditional_data_size, ))
        generated = self.generator([z, c])

        self.discriminator.trainable = False
        valid = self.discriminator([generated, c])

        self.combined = Model([z, c], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(self.learning_rate))

    def bootstrap_generator(self, x, y, epochs, batch_size=128):
        for epoch in range(epochs * (len(x) // batch_size)):
            idx = np.random.randint(0, len(x), batch_size)
            real_data, c = y[idx], x[idx]
            z = np.random.normal(0, 1, (batch_size, self.sequence_length, self.generator_latent_dim))
            loss = self.generator.train_on_batch([z, c], real_data)
            print("Generator bootstrap loss: {}".format(loss))

    def train(self, x, condition, epochs, batch_size=128, sample_interval=100, training_ratio=1):
        valid = np.ones((batch_size, self.sequence_length, 1))
        fake = np.zeros((batch_size, self.sequence_length, 1))

        for epoch in range(epochs * (len(x) // batch_size)):
            # Train Discriminator
            for d_run in range(training_ratio):
                idx = np.random.randint(0, len(x), batch_size)
                real_data, c = x[idx], condition[idx]
                z = np.random.normal(0, 1, (batch_size, self.sequence_length, self.generator_latent_dim))
                generated = self.generator.predict([z, c])

                d_loss_real = self.discriminator.train_on_batch([real_data, c], valid)
                d_loss_fake = self.discriminator.train_on_batch([generated, c], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch([z, c], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    def sample(self, condition):
        z = np.random.normal(0, 1, (len(condition), self.sequence_length, self.generator_latent_dim))
        return self.generator.predict([z, condition])

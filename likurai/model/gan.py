"""
This file will contain objects to create various forms of GANs
"""
import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Input, Concatenate, Dense, TimeDistributed, RepeatVector, Embedding, Multiply, Flatten, BatchNormalization
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop
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

            self.pred = K.placeholder(shape=(None, self.sequence_length, self.n_classes), name='pred')
            self.rewards = K.placeholder(shape=(None, ), name='rewards')

            sequence_input = Input((sequence_length, ))
            conditional_input = Input((1,))

            # TODO: Not sure how I feel about these embedding layers, but let's try them for now
            sequence_embedding = Embedding(self.n_classes+1, hidden_size, input_length=sequence_length, mask_zero=True)(sequence_input)
            conditional_embedding = Embedding(self.conditional_data_size, hidden_size, input_length=1)(conditional_input)
            h = Multiply()([sequence_embedding, conditional_embedding])
            # Stateful = True should allow MCTS style prediction (need to manually reset states each batch during train)
            h = LSTM(hidden_size, return_sequences=False, activation='tanh')(h)

            # Not time distributed since we're doing MCTS style prediction
            logits = Dense(n_classes, activation='softmax')(h)
            self.pg_model = Model(inputs=[sequence_input, conditional_input], outputs=logits)
            self.pg_opt = RMSprop(learning_rate)
            # TODO: Create MLE loss as well. That should be much more straight forward

            # TODO: This doesn't work because it's disconnected from the model. I don't know how to implement rollouts so it isn't though
            # Actually, this is probably why the PyTorch implementation is so simple...since it's eager you can just calculate the loss and do a backward pass manually. No graph bullshit...
            self.pg_loss = -K.sum(
                K.sum(K.one_hot(K.cast(self.pred, np.int32), self.n_classes + 1) * K.log(self.pred)) * self.rewards
            )
            update = self.pg_opt.get_updates(params=self.pg_model.trainable_weights, loss=self.pg_loss)

            self.pg_train = K.function([self.pred, self.rewards], [self.pg_loss], updates=update)
            self.pg_model.summary()

        def do_rollout(self, sequence, conditional_data):
            """
            Takes a batch-size = 1 of sequences and completes them (it) by randomly sampling from the model output.
            In this sense, it is doing an MCTS rollout with its own policy.

            # TODO: Consider using target model (i.e., slowly trailing G) for stability

            :param sequence: For an empty sequence, this will be [[[0]]] (i.e., (1, 1, 1)
            :param conditional_data:
            :return: completed sequence (1, sequence_length, 1)
            """
            if len(sequence.shape) == 2:
                sequence = np.array([sequence])

            while sequence.shape[1] < self.sequence_length:
                logits = self.pg_model.predict([sequence, conditional_data])
                chosen_action = tf.multinomial(logits, 1)
                sequence = np.append(sequence, chosen_action, axis=1)
            return sequence

        # def sample(self, num_samples, sequence_input, conditional_data):
        #     """
        #     Generate num_samples sequences given the input data
        #     :param num_samples:
        #     :param conditional_data:
        #     :return:
        #     """
        #     # inp = np.zeros((num_samples, self.sequence_length, self.n_classes))
        #     output = np.tile(sequence_input, (num_samples, 1, 1))
        #     for i in range(num_samples):
        #         for j in range(self.sequence_length):
        #             if output[i, j].sum() > 0:
        #                 continue
        #             out = self.pg_model.predict_on_batch([output[i], conditional_data.reshape((1, -1))])
        #             choice = np.random.choice(range(self.n_classes), size=1, p=out[j])
        #             output[i, j, choice] = 1
        #     return output

    class Discriminator:

        def __init__(self, n_classes, conditional_data_size, embedding_dim, hidden_size, learning_rate,
                     sequence_length):
            self.n_classes = n_classes
            self.conditional_data_size = conditional_data_size
            self.embedding_dim = embedding_dim
            self.hidden_size = hidden_size
            self.learning_rate = learning_rate
            self.sequence_length = sequence_length

            sequence_input = Input((self.sequence_length, 1))
            conditional_input = Input((1,))

            # TODO: Not sure how I feel about these embedding layers, but let's try them for now
            sequence_embedding = Embedding(self.n_classes + 1, hidden_size, input_length=1, mask_zero=True)(sequence_input)
            conditional_embedding = Embedding(self.conditional_data_size, hidden_size, input_length=1)(conditional_input)
            h = Multiply()([sequence_embedding, conditional_embedding])
            # Stateful = True should allow MCTS style prediction (need to manually reset states each batch during train)
            h = LSTM(hidden_size, return_sequences=False, activation='tanh')(h)

            # Not time distributed since we're doing MCTS style prediction
            logits = Dense(1, activation='sigmoid')(h)
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
        start_sequence = np.zeros((1, 18))

        for epoch in range(epochs):
            # Sample a batch of data
            idxs = np.random.randint(0, len(conditional_data), batch_size)
            # Train generator
            g_loss = 0
            for idx in idxs:
                rollouts = []
                rewards = []
                generated_sequence = self.generator.do_rollout(start_sequence, conditional_data[idx])[0]  # (sequence_length, 1)
                for timestep in range(1, self.sequence_length):
                    rollout = self.generator.do_rollout(generated_sequence[:timestep], conditional_data[idx])[0]
                    reward = self.discriminator.predict(rollout, conditional_data[idx])
                    rollouts.append(rollout)
                    rewards.append(reward)
                pg_loss = self.generator.pg_train(generated_sequence, np.array(rewards))
                print("PG loss: {}".format(pg_loss))

            generated_samples = [self.generator.do_rollout(start_sequence, conditional_data[idx])[0] for idx in idxs]
            real_samples = sequence_data[idxs]
            conditional_samples = conditional_data[idxs]
            d_loss_real = self.discriminator.train_on_batch(generated_samples, conditional_samples, fake)
            d_loss_fake = self.discriminator.train_on_batch(real_samples, conditional_samples, valid)

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

    # TODO: Should these really be time distributed or a) an LSTM(1) b) a single Dense layer (for Disc anyway)
    # TODO: Is it feasible to implement a rollout for the generator here?

    def build_discriminator(self):
        inputs = Input(shape=(self.sequence_length, 1))
        # c = Input(shape=(self.conditional_data_size, ))
        c = Input(shape=(1, ))
        embedding = Embedding(self.conditional_data_size, self.generator_latent_dim)(c)
        embedding = Flatten()(embedding)
        c_stacked = RepeatVector(self.sequence_length)(embedding)
        h = Multiply()([inputs, c_stacked])
        for _ in range(self.n_discriminator_layers):
            h = LSTM(self.discriminator_hidden_size, return_sequences=True, activation='tanh')(h)
            # h = BatchNormalization()(h)
        output = TimeDistributed(Dense(1, activation='sigmoid'))(h)
        model = Model(inputs=[inputs, c], outputs=output)
        print("Discriminator summary...")
        model.summary()
        return model

    def build_generator(self):
        z = Input(shape=(self.sequence_length, self.generator_latent_dim))
        # c = Input(shape=(self.conditional_data_size, ))
        c = Input(shape=(1,))
        embedding = Embedding(self.conditional_data_size, self.generator_latent_dim, input_length=1)(c)
        embedding = Flatten()(embedding)
        c_stacked = RepeatVector(self.sequence_length)(embedding)
        h = Multiply()([z, c_stacked])
        for _ in range(self.n_generator_layers):
            h = LSTM(self.generator_hidden_size, return_sequences=True, activation='tanh')(h)
            # h = BatchNormalization()(h)

        outputs = TimeDistributed(Dense(1, activation='tanh'))(h)
        model = Model(inputs=[z, c], outputs=outputs)
        print("Generator summary...")
        model.summary()
        return model

    def build_models(self):
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.discriminator.compile(RMSprop(self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        self.generator.compile(RMSprop(self.learning_rate), loss='mean_squared_error')

        z = Input(shape=(self.sequence_length, self.generator_latent_dim))
        c = Input(shape=(1, ))
        generated = self.generator([z, c])

        self.discriminator.trainable = False
        valid = self.discriminator([generated, c])

        self.combined = Model([z, c], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=RMSprop(self.learning_rate))

    def bootstrap_generator(self, x, y, epochs, batch_size=128):
        # TODO: This should train the generator to predict the next token (i.e., standard RNN training)
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

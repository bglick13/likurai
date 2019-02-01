"""
This file will contain objects to create various forms of GANs
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import math
import torch.nn.init as init


import numpy as np
import os


class Discriminator(nn.Module):
    def __init__(self, n_classes, conditional_data_size, embedding_dim, hidden_size, sequence_length, n_hidden=2,
               dropout=0.2):
        super(Discriminator , self).__init__()
        self.n_classes = n_classes
        self.conditional_data_sie = conditional_data_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.dropout = dropout

        self.sequence_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        self.conditional_embeddding = nn.Embedding(self.conditional_data_sie, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, num_layers=self.n_hidden, bidirectional=True, dropout=self.dropout)
        self.hidden = nn.Linear(self.hidden_size*2*self.n_hidden, self.hidden_size)
        self.dropout_hidden_to_out = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.hidden_size, 1)

    def forward(self, sequence_input, conditional_input, hidden):
        conditional_input = torch.LongTensor(conditional_input).cuda().unsqueeze(1)

        # sequence_input = sequence_input.permute(1, 0, 2)

        sequence_emb = self.sequence_embedding(sequence_input)
        conditional_emb = self.conditional_embeddding(conditional_input)
        emb = torch.mul(sequence_emb, conditional_emb)
        # emb = emb.view(1, -1, self.embedding_dim)
        emb = emb.permute(1, 0, 2)
        _, hidden = self.gru(emb, hidden)
        # hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.hidden(hidden.view(-1, self.n_hidden*2*self.hidden_size))
        out = torch.tanh(out)
        out = self.dropout_hidden_to_out(out)
        out = self.out(out)
        out = torch.sigmoid(out)
        return out

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(self.n_hidden*2, batch_size, self.hidden_size))
        return h.cuda()

    def predict_on_batch(self, sequence_input, conditional_input, batch_size=1):
        h = self.init_hidden(batch_size)
        out = self.forward(sequence_input, conditional_input, h)
        return out.view(-1)

    def loss(self, pred, true):
        loss_fn = nn.BCELoss()
        return loss_fn(pred, true)


class Generator(nn.Module):
    def __init__(self, n_classes, conditional_data_size, embedding_dim, hidden_size, sequence_length):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.conditional_data_size = conditional_data_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.sequence_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        self.conditional_embeddding = nn.Embedding(self.conditional_data_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.n_classes)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size))
        return h.cuda()

    def forward(self, sequence_input, conditional_input, hidden):
        # sequence_input = torch.LongTensor(sequence_input)
        # conditional_input = torch.LongTensor(conditional_input)
        #
        # sequence_input = sequence_input.cuda()
        # conditional_input = conditional_input.cuda()

        sequence_emb = self.sequence_embedding(sequence_input)
        conditional_emb = self.conditional_embeddding(conditional_input)

        emb = torch.mul(sequence_emb, conditional_emb)
        emb = emb.view(1, -1, self.embedding_dim)
        emb = emb.permute(1, 0, 2)
        out, hidden = self.gru(emb, hidden)  # input (seq_len, batch_size, n_features) --> (seq_len, batch_size, hidden)
        out = out.view(-1, self.hidden_size)
        out = self.out(out)
        out = F.log_softmax(out, dim=1)
        return out, hidden

    # TODO: This
    def sample(self, conditional_input, batch_size, n_samples=1):
        """
        Creates n_samples full sequences from scratch - basically the predict function
        :param conditional_input: (batch_size, conditional_data_size)
        :return: (batch_size, 18, 1)
        """
        samples = torch.zeros(batch_size, self.sequence_length).type(torch.LongTensor).cuda()
        conditional_input = torch.LongTensor(conditional_input).cuda().unsqueeze(1)
        h = self.init_hidden(1)
        sequence_input = autograd.Variable(torch.Tensor([0]*batch_size)).type(torch.LongTensor).cuda().unsqueeze(1)
        for i in range(self.sequence_length):
            out, h = self.forward(sequence_input, conditional_input, h)
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            sequence_input = out.view(-1).unsqueeze(1)

        return samples

    def do_rollout(self, sequence_input, conditional_input, from_scratch):
        # Make sure the inputs are PyTorch Tensors
        if isinstance(sequence_input, (list, np.ndarray)):
            rollout = torch.LongTensor(sequence_input)
            rollout = rollout.cuda()
            sequence_input = torch.LongTensor(sequence_input).cuda()
        else:
            rollout = sequence_input

        # h = self.init_hidden(sequence_input.size()[0])
        h = self.init_hidden(1)
        # Make inputs batch size of 1
        conditional_input = torch.LongTensor(conditional_input).cuda().unsqueeze(0)
        sequence_input = sequence_input.unsqueeze(0)
        # rollout = torch.cat((rollout, torch.zeros(1, self.max_seq_len - len(sequence_input))))

        if from_scratch:
            n_samples = self.sequence_length + 1
            logits = torch.zeros(n_samples, self.n_classes)
        else:
            n_samples = self.sequence_length

        i = 0
        while rollout.size()[0] < n_samples:  # Adding 1 for the start token
            out, h = self.forward(sequence_input, conditional_input, h)
            if from_scratch:
                logits[i, :] = out.view(-1)
                i += 1
            out = torch.multinomial(torch.exp(out), 1)
            rollout = torch.cat((rollout, out.view(-1)))
            sequence_input = out.view(-1)

        if from_scratch:
            return rollout[1:].unsqueeze(0), logits[:-1].unsqueeze(0)
        return rollout.unsqueeze(0)

    def pg_loss(self, pred, rewards):
        # TODO: I'm not convinced this is right but I want to get everything working first
        return torch.sum(torch.mul(pred, rewards).mul(-1))


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

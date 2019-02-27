"""
This file will contain objects to create various forms of GANs
"""
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class Discriminator(nn.Module):
    def __init__(self, n_classes, embedding_dim, hidden_size, sequence_length, n_hidden, dropout,
                 conditional, conditional_type, conditional_merge_type, layer_type,
                 *args, **kwargs):
        """

        :param n_classes:
        :param embedding_dim:
        :param hidden_size:
        :param sequence_length:
        :param n_hidden:
        :param dropout:
        :param conditional:
        :param conditional_type:
        :param conditional_merge_type:
        :param layer_type:
        :param args: Used to pass in as many conditional arrays as desired. Each array should be shape (N, sequence_length, 1)
        :param kwargs:
        """
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        # self.conditional_data_size = conditional_data_size  # number of unique tokens (size of dictionary)
        self.embedding_dim = embedding_dim  # Output/latent dimension of the embedding
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.conditional = conditional
        self.conditional_type = conditional_type
        self.conditional_merge_type = conditional_merge_type
        self.layer_type = layer_type
        self.hidden_layers = []

        self.sequence_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        if self.conditional:
            if self.conditional_type == 'categorical':
                self.conditional_embeddings = nn.ModuleList(
                    [nn.Embedding(len(np.unique(cd)), self.embedding_dim) for cd in args])

            elif self.conditional_type == 'continuous':
                # TODO: This...if we feel it should still be supported
                self.conditional_embeddding = nn.Linear(self.conditional_data_size, self.embedding_dim)

        if self.layer_type == 'gru':
            gru = nn.GRU(self.embedding_dim, self.hidden_size, num_layers=self.n_hidden, bidirectional=True,
                         dropout=self.dropout, batch_first=True)
            self.hidden_layers = gru
            self.hidden = nn.Linear(self.hidden_size * 2 * self.n_hidden, self.hidden_size)
            self.out = nn.Linear(self.hidden_size, 1)

        elif self.layer_type == 'cnn':
            if self.conditional_merge_type == 'mult':
                c_in = self.embedding_dim
            elif self.conditional_merge_type == 'vstack':
                c_in = self.embedding_dim * (len(args) + 1)
            self.hidden_layers = nn.ModuleList([nn.Conv1d(c_in, kwargs['NUM_FILTERS'], ws) for ws in kwargs['WINDOW_SIZES']])
            self.out = nn.Linear(kwargs['NUM_FILTERS'] * len(kwargs['WINDOW_SIZES']), 1)
        self.dropout_hidden_to_out = nn.Dropout(p=self.dropout)

    def forward(self, sequence_input, hidden, *args):
        """

        :param sequence_input: shape = (batch_size, 18)
        :param hidden:
        :param args:
        :return:
        """
        emb = self.sequence_embedding(sequence_input)
        if self.conditional:
            # Make sure the conditional inputs are type torch.LongTensor
            args = [torch.LongTensor(cd).cuda() if isinstance(cd, (list, np.ndarray)) else cd for cd in args]
            conditional_embs = [self.conditional_embeddings[i](args[i].squeeze(2)) for i in range(len(args))]
            # if self.conditional_type == 'continuous':
            #     conditional_emb = torch.tanh(conditional_emb).unsqueeze(1)
            if self.conditional_merge_type == 'mult':
                for ce in conditional_embs:
                    emb = torch.mul(emb, ce)
            elif self.conditional_merge_type == 'vstack':
                emb = torch.cat((emb, torch.cat(conditional_embs, 2)), 2)
        if self.layer_type == 'gru':
            _, hidden = self.hidden_layers(emb, hidden)
            out = self.hidden(hidden.view(-1, self.n_hidden * 2 * self.hidden_size))
            out = torch.tanh(out)

        elif self.layer_type == 'cnn':
            out = emb.permute(0, 2, 1)
            outs = []
            for conv in self.hidden_layers:
                _out = conv(out).permute(0, 2, 1).max(1)[0]
                _out = torch.tanh(_out)
                outs.append(_out)
            out = torch.cat(outs, 1)
            # out = self.hidden(out)

        out = self.dropout_hidden_to_out(out)
        out = self.out(out)
        out = torch.sigmoid(out)

        return out

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(self.n_hidden*2, batch_size, self.hidden_size))
        return h.cuda()

    def predict_on_batch(self, sequence_input, *args):
        """

        :param sequence_input: autograd.Variable
        :param conditional_input: (batch_size, conditional_data_size)
        :param batch_size:
        :return:
        """
        h = self.init_hidden(sequence_input.size()[0])
        if self.conditional:
            out = self.forward(sequence_input, h, *args)
        else:
            out = self.forward(sequence_input, h)

        return out

    def loss(self, pred, true):
        loss_fn = nn.BCELoss()
        accuracy = (torch.round(pred) == true).float().sum() / pred.size()[0]
        return loss_fn(pred, true), accuracy


class Generator(nn.Module):
    def __init__(self, n_classes, embedding_dim, hidden_size, sequence_length, conditional,
                 start_character, conditional_type, conditional_merge_type, dropout, n_hidden, *args):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.conditional = conditional
        self.start_character = start_character
        self.conditional_type = conditional_type
        self.conditional_merge_type = conditional_merge_type
        self.dropout = dropout
        self.n_hidden = n_hidden

        self.sequence_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        if self.conditional:
            if self.conditional_type == 'categorical':
                self.conditional_embeddings = nn.ModuleList(
                    [nn.Embedding(len(np.unique(cd)), self.embedding_dim) for cd in args])
            elif self.conditional_type == 'continuous':
                # TODO: This, if it's still supported
                self.conditional_embeddding = nn.Linear(self.conditional_data_size, self.embedding_dim)

        if self.conditional_merge_type == 'mult':
            input_size = self.embedding_dim
        elif self.conditional_merge_type == 'stack':
            input_size = self.embedding_dim * (len(args) + 1)
        self.gru = nn.GRU(input_size, self.hidden_size, num_layers=self.n_hidden, batch_first=True)
        self.dropout_hidden_to_out = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.hidden_size, self.n_classes)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(self.n_hidden, batch_size, self.hidden_size))
        return h.cuda()

    def forward(self, sequence_input, hidden, *args):
        """

        :param sequence_input: the current token (batch_size, 1)
        :param hidden: The hidden state (num_layers * num_directions, batch, hidden_size)
        :param conditional_input: the conditional input (batch_size, 1)
        :return:
        """
        emb = self.sequence_embedding(sequence_input)

        if self.conditional:
            args = [torch.LongTensor(cd).cuda() if isinstance(cd, (list, np.ndarray)) else cd for cd in args]
            conditional_embs = [self.conditional_embeddings[i](args[i]) for i in range(len(args))]
            if self.conditional_merge_type == 'mult':
                for ce in conditional_embs:
                    emb = torch.mul(emb, ce)
            elif self.conditional_merge_type == 'stack':
                emb = torch.cat((emb, torch.cat(conditional_embs, 2)), 2)

        out, hidden = self.gru(emb, hidden)  # input (batch_size, seq_len, n_features) --> (seq_len, batch_size, hidden)
        out = self.dropout_hidden_to_out(out)
        out = out.view(-1, self.hidden_size)

        out = self.out(out)
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def train_mle(self, sequence_input, targets, *args):
        args = args[0]
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = sequence_input.size()

        h = self.init_hidden(batch_size)
        loss = 0
        for i in range(self.sequence_length):
            _args = [arg[:, i] for arg in args]
            out, h = self.forward(sequence_input[:, i].unsqueeze(1), h, *_args)
            loss += loss_fn(out, targets[:, i])
        return loss

    def sample(self, batch_size, *args):
        """
        Creates n_samples full sequences from scratch - basically the predict function
        Critically, this calculates/processes one sequence at a time and maintains hidden states through each call to forward
        :param conditional_input: (batch_size, conditional_data_size)
        :return: (batch_size, 18, 1)
        """
        samples = torch.zeros(batch_size, self.sequence_length).type(torch.LongTensor).cuda()
        logits = torch.zeros(batch_size, self.sequence_length, self.n_classes)
        if self.conditional:
            args = [torch.LongTensor(cd).cuda() if isinstance(cd, (list, np.ndarray)) else cd for cd in args]
        h = self.init_hidden(batch_size)
        sequence_input = autograd.Variable(torch.Tensor([self.start_character]*batch_size)).type(torch.LongTensor).cuda().unsqueeze(1)
        for i in range(self.sequence_length):
            if self.conditional:
                _args = [arg[:, i] for arg in args]  # arg shape = (batch_size, 1)
                out, h = self.forward(sequence_input, h, *_args)
            else:
                out, h = self.forward(sequence_input, h)

            logits[:, i] = out  # out shape = (batch_size, n_classes)
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            sequence_input = out.view(-1).unsqueeze(1)

        return samples, logits

    def do_rollout(self, sequence_input, *args):
        # Make sure the inputs are PyTorch Tensors
        if isinstance(sequence_input, (list, np.ndarray)):
            sequence_input = torch.Tensor(sequence_input).cuda()

        h = self.init_hidden(1)
        if self.conditional:
            args = [torch.LongTensor(cd).cuda() for cd in args]
        sequence_input = sequence_input.unsqueeze(0)

        starting_length = sequence_input.size()[1]
        rollout = torch.cat((torch.ones(1, 1).type(torch.LongTensor).cuda() * self.start_character,
                             sequence_input.type(torch.LongTensor).cuda(),
                             torch.zeros(1, self.sequence_length-starting_length).type(torch.LongTensor).cuda()), 1)
        for i in range(self.sequence_length):
            if self.conditional:
                _args = [arg[:, i] for arg in args]  # arg shape = (batch_size, 1)
                out, h = self.forward(rollout[:, i].unsqueeze(0), h, *_args)
            else:
                out, h = self.forward(rollout[0, i], h)

            out = torch.multinomial(torch.exp(out), 1)

            if i >= starting_length:
                rollout[0, i+1] = out.view(-1)

        return rollout[0, 1:]

    def pg_loss(self, pred, rewards):
        return torch.sum(torch.mul(pred, rewards).mul(-1)) / pred.size()[0]


class SeqGAN:
    def __init__(self, generator, discriminator, g_opt, d_opt, model_name='model'):
        self.generator = generator
        self.discriminator = discriminator
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.model_name = model_name

        self.start_character = generator.start_character
        self.sequence_length = generator.sequence_length
        self.n_classes = generator.n_classes
        self.conditional = self.generator.conditional

    def create_supervised_batch(self, sequences, *args):
        batch_size = len(sequences)
        sequence_inputs = torch.zeros(batch_size, self.sequence_length)
        sequence_inputs[:, 0] = self.start_character
        sequence_inputs[:, 1:] = torch.Tensor(sequences[:, :self.sequence_length - 1])
        sequence_inputs = Variable(sequence_inputs).type(torch.LongTensor).cuda()
        args = [torch.LongTensor(cd).cuda() for cd in args]
        targets = Variable(torch.Tensor(sequences)).type(torch.LongTensor).cuda()
        return sequence_inputs, targets, args

    def pretrain_generator(self, sequence_train, epochs, batch_size, *args):
        for epoch in range(epochs):
            idxs = np.random.randint(0, len(sequence_train), batch_size)
            self.g_opt.zero_grad()
            _args = [arg[idxs] for arg in args]
            seq_x, targets, con_x = self.create_supervised_batch(sequence_train[idxs], *_args)
            mle_loss = self.generator.train_mle(seq_x, targets, con_x)

            mle_loss.backward()
            self.g_opt.step()
            print("Epoch [{}] MLE Loss: {}".format(epoch, mle_loss))

    def pretrain_discriminator(self, sequence_train, epochs, batch_size, *args):
        if self.conditional:
            args = [torch.LongTensor(cd).cuda() for cd in args]
        for epoch in range(epochs):
            idxs = np.random.randint(0, len(sequence_train), batch_size)

            # Get the data we're training on
            _args = [arg[idxs] for arg in args]
            fake_pretrain, _ = self.generator.sample(batch_size, *_args)
            real_pretrain = Variable(torch.Tensor(sequence_train[idxs]).type(torch.LongTensor)).cuda()

            # Generate predictions
            for _input, _target in zip([real_pretrain, fake_pretrain],
                                       [torch.ones(batch_size, 1).cuda(), torch.zeros(batch_size, 1).cuda()]):
                if self.conditional:
                    pred = self.discriminator.predict_on_batch(_input, *_args)
                else:
                    pred = self.discriminator.predict_on_batch(_input)

                # Backprop the loss
                self.d_opt.zero_grad()
                loss, accuracy = self.discriminator.loss(pred, _target)  # true shape = (batch_szie, 1)
                loss.backward()
                self.d_opt.step()
                print("Epoch [{}] Discriminator Loss: {:.3f}, Accuracy: {:.3f}".format(epoch, loss, accuracy))
            print('\n')

    def calculate_rewards(self, sequence_input, n_rollouts=1, *args):
        if isinstance(sequence_input, (list, np.ndarray)):
            sequence_input = torch.Tensor(sequence_input).type(torch.LongTensor).cuda()

        rewards = torch.zeros(self.sequence_length, self.n_classes)
        for timestep in range(self.sequence_length):
            reward = 0
            for n in range(n_rollouts):
                rollout = self.generator.do_rollout(sequence_input[0, :timestep + 1], *args).unsqueeze(0)
                reward += self.discriminator.predict_on_batch(rollout, *args)
            rewards[timestep, sequence_input.view(-1)[timestep]] = reward / n_rollouts
        return rewards

    def _adversarial_discriminator_update(self, sequence_train, batch_size, *args):
        valid = torch.ones((batch_size, 1)).cuda()
        fake = torch.zeros((batch_size, 1)).cuda()
        idxs = np.random.randint(0, len(sequence_train), batch_size)
        real_samples = torch.Tensor(sequence_train[idxs]).type(torch.LongTensor).cuda()

        if self.conditional:
            args = [arg[idxs] for arg in args]
            generated_samples, _ = self.generator.sample(batch_size, *args)
            d_pred_fake = self.discriminator.predict_on_batch(generated_samples, *args)
            d_pred_real = self.discriminator.predict_on_batch(real_samples, *args)
        else:
            generated_samples, _ = self.generator.sample(batch_size)
            d_pred_fake = self.discriminator.predict_on_batch(generated_samples)
            d_pred_real = self.discriminator.predict_on_batch(real_samples)

        total_d_loss = 0
        total_accuracy = 0
        for _input, _target in zip([d_pred_real, d_pred_fake],
                                   [valid, fake]):
            self.d_opt.zero_grad()
            d_loss, accuracy = self.discriminator.loss(_input, _target)
            d_loss.backward()
            self.d_opt.step()

            total_d_loss += d_loss
            total_accuracy += accuracy
        return total_d_loss/2., total_accuracy/2.

    def _adversarial_generator_update(self, n_rollouts=1, *args):
        # 1) Generate a sequence
        if self.conditional:
            # con_input = np.array([conditional_train])
            generated_sequence, logits = self.generator.sample(1, *args)
        else:
            generated_sequence, logits = self.generator.sample(1)
            con_input = None

        # 2) For each timestep, calculate the reward
        rewards = self.calculate_rewards(generated_sequence, n_rollouts, *args).unsqueeze(0)

        # 3) Update generator
        self.g_opt.zero_grad()
        pg_loss = self.generator.pg_loss(logits, rewards)
        pg_loss.backward()
        self.g_opt.step()
        return pg_loss

    def evaluate_test_set(self, sequence_test, n_samples=1, epoch=1, out_dir='', *args):
        # These are shape (n_samples * len(sequence_test), sequence_length)
        generated_samples = np.zeros((len(sequence_test), self.sequence_length))
        generated_scores = np.zeros((len(sequence_test), 1))
        for i, seq in enumerate(sequence_test):
            con = [np.expand_dims(cd[i], 0).repeat(n_samples, 0) for cd in args]
            _generated_samples, _ = self.generator.sample(n_samples, *con)
            _generated_scores = self.discriminator.predict_on_batch(_generated_samples, *con)
            generated_samples[i] = _generated_samples[_generated_scores.argmax()].detach().cpu().numpy()
            generated_scores[i] = _generated_scores.max().detach().cpu().numpy()

        # cluster = TSNE().fit_transform(_emb.view(_emb.size()[0] * self.sequence_length, self.discriminator.embedding_dim).detach().cpu().numpy())
        # c = _generated_samples.view(_emb.size()[0] * self.sequence_length).cpu().numpy()
        # fig, ax = plt.subplots(figsize=(17, 8))
        # ax.scatter(cluster[:, 0], cluster[:, 1], c=c)
        # plt.colorbar(fig)
        # plt.savefig('{}/embedding_{}.png'.format(out_dir, epoch))
        # plt.close()

        generated_to_par = pd.DataFrame(generated_samples - 3)
        generated_dk = generated_to_par.replace({-3: 20, -2: 8, -1: 3, 0: .5, 1: -.5, 2: -1})
        test_to_par = pd.DataFrame(sequence_test - 3)
        test_dk = test_to_par.replace({-3: 20, -2: 8, -1: 3, 0: .5, 1: -.5, 2: -1})

        r2 = r2_score(test_to_par.sum(1), generated_to_par.sum(1))
        dk_r2 = r2_score(test_dk.sum(1), generated_dk.sum(1))

        print("R2/DK: {}/{}".format(r2, dk_r2))
        sns.jointplot(generated_to_par.sum(1), test_to_par.sum(1))
        plt.title('Epoch [{}] r2: {}'.format(epoch, r2))
        plt.savefig('{}/pred_iter_{}.png'.format(out_dir, epoch))

        sns.jointplot(generated_dk.sum(1), test_dk.sum(1))
        plt.title('Epoch [{}] r2: {}'.format(epoch, r2))
        plt.savefig('{}/dk_pred_iter_{}.png'.format(out_dir, epoch))

        con = [np.expand_dims(cd[0], 0) for cd in args]
        sample_reward = self.calculate_rewards(generated_samples[0].reshape(1, -1), 25, *con).detach().cpu().numpy().squeeze().max(1)
        out = pd.DataFrame({'fake': generated_to_par.loc[0, :], 'real': test_to_par.loc[0, :], 'reward': sample_reward})

        fig, ax = plt.subplots(figsize=(17, 8))
        sns.distplot(generated_scores, ax=ax)
        plt.title("Epoch [{}] discriminator predictions".format(epoch))
        plt.savefig('{}/disc_pred_{}.png'.format(out_dir, epoch))
        plt.close()
        print(out)

    def train_adversarial(self, sequence_train, epochs, batch_size, sequence_test, conditional_train=None,
                          conditional_test=None, n_g_steps=1, n_rollouts=1, training_ratio=1, n_eval_samples=1, out_dir=''):
        for epoch in range(epochs+1):
            # Sample a batch of data
            idxs = np.random.randint(0, len(sequence_train), n_g_steps)
            # Train generator
            for i, idx in enumerate(idxs):
                _args = [np.expand_dims(arg[idx], 0) for arg in conditional_train]
                pg_loss = self._adversarial_generator_update(n_rollouts, *_args)
                print("Epoch [{}], PG Loss: {}".format(epoch, pg_loss))

            # 4) Generate sequences to train discriminator with
            for _ in range(training_ratio):
                d_loss, d_accuracy = self._adversarial_discriminator_update(sequence_train, batch_size, *conditional_train)
                print("%d [D loss: %f, accuracy: %f] [G loss: %f]" % (epoch, d_loss, d_accuracy, pg_loss))

            # 5) Evaluate on test set
            if epoch % 5 == 0:
                self.evaluate_test_set(sequence_test, n_eval_samples, epoch, out_dir, *conditional_test)

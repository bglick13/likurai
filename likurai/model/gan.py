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
    def __init__(self, n_classes, conditional_data_size, embedding_dim, hidden_size, sequence_length, n_hidden=1,
               dropout=0.2, conditional=False, conditional_type='categorical', layer_type='cnn', **kwargs):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.conditional_data_size = conditional_data_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.conditional = conditional
        self.conditional_type = conditional_type
        self.layer_type = layer_type
        self.hidden_layers = []

        self.sequence_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        if self.conditional:
            if self.conditional_type == 'categorical':
                self.conditional_embeddding = nn.Embedding(self.conditional_data_size, self.embedding_dim)
            elif self.conditional_type == 'continuous':
                self.conditional_embeddding = nn.Linear(self.conditional_data_size, self.embedding_dim)

        if self.layer_type == 'gru':
            gru = nn.GRU(self.embedding_dim, self.hidden_size, num_layers=self.n_hidden, bidirectional=True,
                         dropout=self.dropout, batch_first=True)
            self.hidden_layers = gru
            self.hidden = nn.Linear(self.hidden_size * 2 * self.n_hidden, self.hidden_size)
            self.out = nn.Linear(self.hidden_size, 1)

        elif self.layer_type == 'cnn':
            self.hidden_layers = nn.ModuleList([nn.Conv1d(self.embedding_dim, kwargs['NUM_FILTERS'], ws) for ws in kwargs['WINDOW_SIZES']])
            self.out = nn.Linear(kwargs['NUM_FILTERS'] * len(kwargs['WINDOW_SIZES']), 1)
        self.dropout_hidden_to_out = nn.Dropout(p=self.dropout)

    def forward(self, sequence_input, hidden, conditional_input=None, ret_emb=False):
        if self.conditional:
            if isinstance(conditional_input, (list, np.ndarray)):
                conditional_input = torch.LongTensor(conditional_input).cuda()

        sequence_emb = self.sequence_embedding(sequence_input)
        # if visualize:
        #     cluster = TSNE().fit_transform(sequence_emb.view(sequence_emb.size()[0] * self.sequence_length, self.embedding_dim).detach().cpu().numpy())
        #     scat = plt.scatter(cluster[:, 0], cluster[:, 1], c=sequence_input.view(sequence_emb.size()[0] * self.sequence_length).cpu().numpy())
        #     plt.colorbar(scat)
        #     plt.savefig('tmp.png')
        # else:
        #     scat = None

        if self.conditional:
            conditional_emb = self.conditional_embeddding(conditional_input)
            if self.conditional_type == 'continuous':
                conditional_emb = torch.tanh(conditional_emb).unsqueeze(1)
            emb = torch.mul(sequence_emb, conditional_emb)
        else:
            emb = sequence_emb

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
        if ret_emb:
            return out, sequence_emb
        return out

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(self.n_hidden*2, batch_size, self.hidden_size))
        return h.cuda()

    def predict_on_batch(self, sequence_input, conditional_input=None, ret_emb=False):
        """

        :param sequence_input: autograd.Variable
        :param conditional_input: (batch_size, conditional_data_size)
        :param batch_size:
        :return:
        """
        h = self.init_hidden(sequence_input.size()[0])
        if self.conditional:
            out = self.forward(sequence_input, h, conditional_input, ret_emb=ret_emb)
        else:
            out = self.forward(sequence_input, h)

        return out

    def loss(self, pred, true):
        loss_fn = nn.BCELoss()
        accuracy = (torch.round(pred) == true).float().sum() / pred.size()[0]
        return loss_fn(pred, true), accuracy


class Generator(nn.Module):
    def __init__(self, n_classes, conditional_data_size, embedding_dim, hidden_size, sequence_length, conditional=False,
                 start_character=3, conditional_type='categorical', dropout=0.0, n_hidden=1):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.conditional_data_size = conditional_data_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.conditional = conditional
        self.start_character = start_character
        self.conditional_type = conditional_type
        self.dropout = dropout
        self.n_hidden = n_hidden

        self.sequence_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        if self.conditional:
            if self.conditional_type == 'categorical':
                self.conditional_embeddding = nn.Embedding(self.conditional_data_size, self.embedding_dim)
            elif self.conditional_type == 'continuous':
                self.conditional_embeddding = nn.Linear(self.conditional_data_size, self.embedding_dim)
        # self.gru = nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, num_layers=self.n_hidden, batch_first=True)
        self.dropout_hidden_to_out = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.hidden_size, self.n_classes)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(self.n_hidden, batch_size, self.hidden_size))
        return h.cuda()

    def forward(self, sequence_input, hidden, conditional_input=None):
        """

        :param sequence_input: the current token (batch_size, 1)
        :param hidden: The hidden state (num_layers * num_directions, batch, hidden_size)
        :param conditional_input: the conditional input (batch_size, 1)
        :return:
        """
        sequence_emb = self.sequence_embedding(sequence_input)

        if self.conditional:
            conditional_emb = self.conditional_embeddding(conditional_input)
            if self.conditional_type == 'continuous':
                conditional_emb = torch.tanh(conditional_emb).unsqueeze(1)
            emb = torch.mul(sequence_emb, conditional_emb)
        else:
            emb = sequence_emb

        out, hidden = self.gru(emb, hidden)  # input (seq_len, batch_size, n_features) --> (seq_len, batch_size, hidden)
        out = self.dropout_hidden_to_out(out)
        out = out.view(-1, self.hidden_size)

        out = self.out(out)
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def train_mle(self, sequence_input, targets, conditional_input=None):
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = sequence_input.size()

        h = self.init_hidden(batch_size)
        loss = 0
        for i in range(self.sequence_length):
            out, h = self.forward(sequence_input[:, i].unsqueeze(1), h, conditional_input)
            loss += loss_fn(out, targets[:, i])
        return loss

    def sample(self, batch_size, conditional_input=None):
        """
        Creates n_samples full sequences from scratch - basically the predict function
        Critically, this calculates/processes one sequence at a time and maintains hidden states through each call to forward
        :param conditional_input: (batch_size, conditional_data_size)
        :return: (batch_size, 18, 1)
        """
        samples = torch.zeros(batch_size, self.sequence_length).type(torch.LongTensor).cuda()
        logits = torch.zeros(batch_size, self.sequence_length, self.n_classes)
        if self.conditional:
            if isinstance(conditional_input, (list, np.ndarray)):
                conditional_input = torch.LongTensor(conditional_input).cuda()  #.unsqueeze(1)
        h = self.init_hidden(batch_size)
        sequence_input = autograd.Variable(torch.Tensor([self.start_character]*batch_size)).type(torch.LongTensor).cuda().unsqueeze(1)
        for i in range(self.sequence_length):
            if self.conditional:
                out, h = self.forward(sequence_input, h, conditional_input)
            else:
                out, h = self.forward(sequence_input, h)

            logits[:, i] = out
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            sequence_input = out.view(-1).unsqueeze(1)

        return samples, logits

    def do_rollout(self, sequence_input, conditional_input=None):
        # Make sure the inputs are PyTorch Tensors
        if isinstance(sequence_input, (list, np.ndarray)):
            sequence_input = torch.Tensor(sequence_input).cuda()

        h = self.init_hidden(1)
        if self.conditional:
            conditional_input = torch.LongTensor(conditional_input).cuda()
        sequence_input = sequence_input.unsqueeze(0)

        starting_length = sequence_input.size()[1]
        rollout = torch.cat((torch.ones(1, 1).type(torch.LongTensor).cuda() * self.start_character,
                             sequence_input.type(torch.LongTensor).cuda(),
                             torch.zeros(1, self.sequence_length-starting_length).type(torch.LongTensor).cuda()), 1)
        for i in range(self.sequence_length):
            if self.conditional:
                out, h = self.forward(rollout[0, i], h, conditional_input)
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

    def create_supervised_batch(self, sequences, conditionals=None):
        batch_size = len(sequences)
        sequence_inputs = torch.zeros(batch_size, self.sequence_length)
        sequence_inputs[:, 0] = self.start_character
        sequence_inputs[:, 1:] = torch.Tensor(sequences[:, :self.sequence_length - 1])
        sequence_inputs = Variable(sequence_inputs).type(torch.LongTensor).cuda()
        conditionals = Variable(torch.LongTensor(conditionals)).cuda()
        targets = Variable(torch.Tensor(sequences)).type(torch.LongTensor).cuda()
        return sequence_inputs, targets, conditionals

    def pretrain_generator(self, sequence_train, epochs, batch_size, conditional_train=None):
        for epoch in range(epochs):
            idxs = np.random.randint(0, len(sequence_train), batch_size)
            self.g_opt.zero_grad()

            seq_x, targets, con_x = self.create_supervised_batch(sequence_train[idxs], conditional_train[idxs])
            mle_loss = self.generator.train_mle(seq_x, targets, con_x)

            mle_loss.backward()
            self.g_opt.step()
            print("Epoch [{}] MLE Loss: {}".format(epoch, mle_loss))

    def pretrain_discriminator(self, sequence_train, epochs, batch_size, conditional_train=None):
        if conditional_train is not None:
            conditional_train = torch.LongTensor(conditional_train).cuda()
        for epoch in range(epochs):
            idxs = np.random.randint(0, len(sequence_train), batch_size)

            # Get the data we're training on

            fake_pretrain, _ = self.generator.sample(batch_size, conditional_train[idxs])
            real_pretrain = Variable(torch.Tensor(sequence_train[idxs]).type(torch.LongTensor)).cuda()

            # Generate predictions
            for _input, _target in zip([real_pretrain, fake_pretrain],
                                       [torch.ones(batch_size, 1).cuda(), torch.zeros(batch_size, 1).cuda()]):
                if conditional_train is not None:
                    pred = self.discriminator.predict_on_batch(_input, conditional_input=conditional_train[idxs])
                else:
                    pred = self.discriminator.predict_on_batch(_input)

                # Backprop the loss
                self.d_opt.zero_grad()
                loss, accuracy = self.discriminator.loss(pred, _target)  # true shape = (batch_szie, 1)
                loss.backward()
                self.d_opt.step()
                print("Epoch [{}] Discriminator Loss: {:.3f}, Accuracy: {:.3f}".format(epoch, loss, accuracy))
            print('\n')

    def calculate_rewards(self, sequence_input, conditional_input=None, n_rollouts=1):
        if isinstance(sequence_input, (list, np.ndarray)):
            sequence_input = torch.Tensor(sequence_input).type(torch.LongTensor).cuda()

        rewards = torch.zeros(self.sequence_length, self.n_classes)
        for timestep in range(self.sequence_length):
            reward = 0
            for n in range(n_rollouts):
                rollout = self.generator.do_rollout(sequence_input[0, :timestep + 1], conditional_input)
                reward += self.discriminator.predict_on_batch(rollout, conditional_input)
            rewards[timestep, sequence_input.view(-1)[timestep]] = reward / n_rollouts
        return rewards

    def _adversarial_discriminator_update(self, sequence_train, batch_size, conditional_train=None):
        valid = torch.ones((batch_size, 1)).cuda()
        fake = torch.zeros((batch_size, 1)).cuda()
        idxs = np.random.randint(0, len(sequence_train), batch_size)
        real_samples = torch.Tensor(sequence_train[idxs]).type(torch.LongTensor).cuda()

        if conditional_train is not None:
            conditional_samples = conditional_train[idxs]
            generated_samples, _ = self.generator.sample(batch_size=batch_size, conditional_input=conditional_samples)
            d_pred_fake = self.discriminator.predict_on_batch(generated_samples, conditional_samples)
            d_pred_real = self.discriminator.predict_on_batch(real_samples, conditional_samples)
        else:
            generated_samples, _ = self.generator.sample(batch_size=batch_size)
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

    def _adversarial_generator_update(self, conditional_train=None, n_rollouts=1):
        # 1) Generate a sequence
        if conditional_train is not None:
            con_input = np.array([conditional_train])
            generated_sequence, logits = self.generator.sample(1, con_input)
        else:
            generated_sequence, logits = self.generator.sample(1)
            con_input = None

        # 2) For each timestep, calculate the reward
        rewards = self.calculate_rewards(generated_sequence, con_input, n_rollouts).unsqueeze(0)

        # 3) Update generator
        self.g_opt.zero_grad()
        pg_loss = self.generator.pg_loss(logits, rewards)
        pg_loss.backward()
        self.g_opt.step()
        return pg_loss

    def evaluate_test_set(self, sequence_test, conditional_test=None, n_samples=1, epoch=1, out_dir=''):
        # These are shape (n_samples * len(sequence_test), sequence_length)
        generated_samples = np.zeros((len(sequence_test), self.sequence_length))
        generated_scores = np.zeros((len(sequence_test), 1))
        for i, con in enumerate(conditional_test):
            _generated_samples, _ = self.generator.sample(n_samples, con.repeat(n_samples).reshape(-1, 1))
            _generated_scores = self.discriminator.predict_on_batch(_generated_samples, con.repeat(n_samples).reshape(-1, 1), ret_emb=False)
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

        sample_reward = self.calculate_rewards(generated_samples[0].reshape(1, -1),
                                               conditional_test[0].reshape(1, -1), 25).detach().cpu().numpy().squeeze().max(1)
        out = pd.DataFrame({'fake': generated_to_par.loc[0, :], 'real': test_to_par.loc[0, :],
                           'reward': sample_reward})

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
               pg_loss = self._adversarial_generator_update(conditional_train[idx], n_rollouts)
               print("Epoch [{}], PG Loss: {}".format(epoch, pg_loss))

            # 4) Generate sequences to train discriminator with
            for _ in range(training_ratio):
                d_loss, d_accuracy = self._adversarial_discriminator_update(sequence_train, batch_size, conditional_train)
                print("%d [D loss: %f, accuracy: %f] [G loss: %f]" % (epoch, d_loss, d_accuracy, pg_loss))

            # 5) Evaluate on test set
            if epoch % 5 == 0:
                self.evaluate_test_set(sequence_test, conditional_test, n_eval_samples, out_dir=out_dir, epoch=epoch)

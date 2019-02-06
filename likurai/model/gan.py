"""
This file will contain objects to create various forms of GANs
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Discriminator(nn.Module):
    def __init__(self, n_classes, conditional_data_size, embedding_dim, hidden_size, sequence_length, n_hidden=1,
               dropout=0.2, conditional=False, conditional_type='categorical'):
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

        self.sequence_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        if self.conditional:
            if self.conditional_type == 'categorical':
                self.conditional_embeddding = nn.Embedding(self.conditional_data_size, self.embedding_dim)
            elif self.conditional_type == 'continuous':
                self.conditional_embeddding = nn.Linear(self.conditional_data_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, num_layers=self.n_hidden, bidirectional=True, dropout=self.dropout)
        self.hidden = nn.Linear(self.hidden_size*2*self.n_hidden, self.hidden_size)
        self.dropout_hidden_to_out = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(self.hidden_size, 1)

    def forward(self, sequence_input, hidden, conditional_input=None):
        if self.conditional:
            if isinstance(conditional_input, (list, np.ndarray)):
                conditional_input = torch.Tensor(conditional_input).cuda()  #.unsqueeze(1)

        sequence_emb = self.sequence_embedding(sequence_input)
        if self.conditional:
            conditional_emb = self.conditional_embeddding(conditional_input)
            if self.conditional_type == 'continuous':
                conditional_emb = torch.tanh(conditional_emb).unsqueeze(1)
            emb = torch.mul(sequence_emb, conditional_emb)
        else:
            emb = sequence_emb

        emb = emb.permute(1, 0, 2)
        _, hidden = self.gru(emb, hidden)
        out = self.hidden(hidden.view(-1, self.n_hidden*2*self.hidden_size))
        out = torch.tanh(out)
        out = self.dropout_hidden_to_out(out)
        out = self.out(out)
        out = torch.sigmoid(out)
        return out

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(self.n_hidden*2, batch_size, self.hidden_size))
        return h.cuda()

    def predict_on_batch(self, sequence_input, conditional_input=None, batch_size=1):
        """

        :param sequence_input: autograd.Variable
        :param conditional_input:
        :param batch_size:
        :return:
        """
        h = self.init_hidden(sequence_input.size()[0])
        if self.conditional:
            out = self.forward(sequence_input, h, conditional_input)
        else:
            out = self.forward(sequence_input, h)
        return out.view(-1)

    def loss(self, pred, true):
        loss_fn = nn.BCELoss()
        return loss_fn(pred, true)


class Generator(nn.Module):
    def __init__(self, n_classes, conditional_data_size, embedding_dim, hidden_size, sequence_length, conditional=False,
                 start_character=3, conditional_type='categorical'):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.conditional_data_size = conditional_data_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.conditional = conditional
        self.start_character = start_character
        self.conditional_type = conditional_type

        self.sequence_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        if self.conditional:
            if self.conditional_type == 'categorical':
                self.conditional_embeddding = nn.Embedding(self.conditional_data_size, self.embedding_dim)
            elif self.conditional_type == 'continuous':
                self.conditional_embeddding = nn.Linear(self.conditional_data_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.n_classes)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size))
        return h.cuda()

    def forward(self, sequence_input, hidden, conditional_input=None):
        sequence_emb = self.sequence_embedding(sequence_input)

        if self.conditional:
            conditional_emb = self.conditional_embeddding(conditional_input)
            if self.conditional_type == 'continuous':
                conditional_emb = torch.tanh(conditional_emb).unsqueeze(1)
            emb = torch.mul(sequence_emb, conditional_emb)
        else:
            emb = sequence_emb

        emb = emb.permute(1, 0, 2)
        out, hidden = self.gru(emb, hidden)  # input (seq_len, batch_size, n_features) --> (seq_len, batch_size, hidden)

        out = out.view(-1, self.hidden_size)
        out = self.out(out)
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def train_mle(self, sequence_input, targets, conditional_input=None):
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = sequence_input.size()

        if self.conditional:
            if self.conditional_type == 'categorical':
                conditional_input = conditional_input.unsqueeze(1)

        h = self.init_hidden(batch_size)
        loss = 0
        for i in range(self.sequence_length):
            if self.conditional:
                out, h = self.forward(sequence_input[:, i].unsqueeze(1), h, conditional_input)
            else:
                out, h = self.forward(sequence_input[:, i].unsqueeze(1), h)
            loss += loss_fn(out, targets[:, i])
        return loss

    def sample(self, batch_size, conditional_input=None):
        """
        Creates n_samples full sequences from scratch - basically the predict function
        :param conditional_input: (batch_size, conditional_data_size)
        :return: (batch_size, 18, 1)
        """
        samples = torch.zeros(batch_size, self.sequence_length).type(torch.LongTensor).cuda()
        logits = torch.zeros(batch_size, self.sequence_length, self.n_classes)
        if self.conditional:
            if isinstance(conditional_input, (list, np.ndarray)):
                conditional_input = torch.Tensor(conditional_input).cuda()  #.unsqueeze(1)
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
        # TODO: Let's just make this its own helper function to clean everything up
        if isinstance(sequence_input, (list, np.ndarray)):
            rollout = torch.Tensor(sequence_input).cuda()
            sequence_input = torch.Tensor(sequence_input).cuda()
        else:
            rollout = sequence_input

        h = self.init_hidden(1)
        if self.conditional:
            conditional_input = torch.Tensor(conditional_input).cuda()  #.unsqueeze(0)
        sequence_input = sequence_input.unsqueeze(0)

        while rollout.size()[0] < self.sequence_length:
            if self.conditional:
                out, h = self.forward(sequence_input, h, conditional_input)
            else:
                out, h = self.forward(sequence_input, h)

            out = torch.multinomial(torch.exp(out), 1)
            rollout = torch.cat((rollout, out.view(-1)))
            sequence_input = out.view(-1).unsqueeze(0)

        return rollout.unsqueeze(0)

    def pg_loss(self, pred, rewards):
        # TODO: I'm not convinced this is right but I want to get everything working first
        return torch.sum(torch.mul(pred, rewards).mul(-1)) / pred.size()[0]



import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MultiLayerGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, bias=True):
        super(MultiLayerGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        self.gru_cells = nn.ModuleList([nn.GRUCell(input_size, hidden_size, bias)])
        for _ in range(num_layers - 1):
            self.gru_cells.append(nn.GRUCell(input_size, hidden_size, bias))

    def forward(self, input, states):
        """
        :param input: FloatTensor (batch_size, time_step, input_size)
        :param states: FloatTensor (num_layers, batch_size, hidden_size)
        :return output_hidden: FloatTensor (num_layers, batch_size, hidden_size)
        """
        hidden = states
        output_hidden = []
        for i, gru_cell in enumerate(self.gru_cells):
            h = gru_cell(input, hidden[i])
            output_hidden.append(h)
            input = F.dropout(h, p=self.dropout, training=self.training)
        output_hidden = torch.stack(output_hidden, dim=0)
        return output_hidden

# Encoder
# ------------------------------------------------------------------------------

# Encode into Z with mu and log_var

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True, cuda=True):
        super(EncoderRNN, self).__init__()
        self.cuda = cuda
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=0.1, bidirectional=bidirectional)
        self.o2p = nn.Linear(hidden_size, output_size * 2)

    def forward(self, input):
        embedded = self.embed(input)
        output, hidden = self.gru(embedded, None)

        output = output[:,-1,:]  # Take only the last value
        if self.bidirectional:
            output = output[:, :self.hidden_size] + output[:,
                                                           self.hidden_size:]  # Sum bidirectional outputs
        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar)
        return mu, logvar, z

    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if self.cuda:
            eps = eps.cuda()
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

# Decoder
# ------------------------------------------------------------------------------

# Decode from Z into sequence


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.1, cuda=True):
        super(DecoderRNN, self).__init__()
        self.SOS = 1
        self.cuda = cuda
        self.embed = nn.Embedding(output_size, input_size)
        self.dropout = nn.Dropout(dropout_p)
        self.output_size = output_size
        self.n_layers = n_layers
        self.rnn_cell = MultiLayerGRUCell(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=n_layers,
            dropout=dropout_p
        )
        self.output_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size)
        )
        self.generator = nn.Linear(input_size, output_size)

    def forward(self, hidden, trg):
        max_len = trg.size(1)
        hidden = hidden.unsqueeze(0).repeat(self.n_layers,1,1)
        logit = []
        input = Variable(torch.LongTensor([self.SOS]*trg.size(0)))
        if self.cuda:
            input = input.cuda()
        for i in range(max_len):
            hidden, input = self.step(hidden, input)
            logit.append(input)
            use_teacher_forcing = random.random() <= 0.9 if self.training and i > 0 else 1
            if use_teacher_forcing:
                input = trg[:,i]
            else:
                input = input.argmax(dim=1)
        logit = torch.stack(logit, dim=1)
        return logit



    def step(self, hidden, token):
        token_embedding = self.embed(token.unsqueeze(0)).squeeze(0)
        hidden = self.rnn_cell(token_embedding, hidden)
        top_hidden = hidden[-1]
        output = self.output_projection(top_hidden)
        token_logit = self.generator(output)
        return hidden, token_logit


# Container
# ------------------------------------------------------------------------------


class VAE(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_size, dropout=0.1,n_layers=1, bidirectional=True, cuda=True):
        super(VAE, self).__init__()
        self.encoder = EncoderRNN(vocab_size, hidden_size, emb_size,
                                  n_layers=n_layers, bidirectional=bidirectional, cuda=cuda)
        self.decoder = DecoderRNN(emb_size, hidden_size, vocab_size,
                                  n_layers=n_layers, dropout_p=dropout,cuda=cuda)

    def forward(self, inputs):
        m, l, z = self.encoder(inputs)
        decoded = self.decoder(z, inputs)
        return m, l, z, decoded

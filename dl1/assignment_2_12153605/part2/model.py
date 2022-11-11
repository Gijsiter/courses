###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Adapted: 2021-11-11
###############################################################################

import math
import torch
import torch.nn as nn
import numpy as np

from torch._C import device


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """
    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Define weight matrices.
        self.Wgx = nn.Parameter(torch.empty((self.hidden_dim, self.embed_dim)))
        self.Wgh = nn.Parameter(torch.empty((self.hidden_dim, self.hidden_dim)))

        self.Wix = nn.Parameter(torch.empty((self.hidden_dim, self.embed_dim)))
        self.Wih = nn.Parameter(torch.empty((self.hidden_dim, self.hidden_dim)))

        self.Wfx = nn.Parameter(torch.empty((self.hidden_dim, self.embed_dim)))
        self.Wfh = nn.Parameter(torch.empty((self.hidden_dim, self.hidden_dim)))

        self.Wox = nn.Parameter(torch.empty((self.hidden_dim, self.embed_dim)))
        self.Woh = nn.Parameter(torch.empty((self.hidden_dim, self.hidden_dim)))
        # Define bias vectors.
        self.bg = nn.Parameter(torch.empty((self.hidden_dim, 1)))
        self.bi = nn.Parameter(torch.empty((self.hidden_dim, 1)))
        self.bf = nn.Parameter(torch.empty((self.hidden_dim, 1)),
                               requires_grad=False)
        self.bo = nn.Parameter(torch.empty((self.hidden_dim, 1)))

        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for param in self.parameters():
            nn.init.uniform_(param, -1 / math.sqrt(self.hidden_dim),
                                     1 / math.sqrt(self.hidden_dim))
        self.bf += 1.
        self.bf.requires_grad = True
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds, ht_1=None, ct_1=None):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, hidden dimension].
            ht_1: hidden state at t-1 (necessary for sampling)
            ct_1: candidate memory at t-1 (necessary for sampling)
        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        device = self.Wfh.device
        out = None
        Wgxh = torch.cat((self.Wgx, self.Wgh), dim=1)
        Wixh = torch.cat((self.Wix, self.Wih), dim=1)
        Wfxh = torch.cat((self.Wfx, self.Wfh), dim=1)
        Woxh = torch.cat((self.Wox, self.Woh), dim=1)

        if ht_1 is None and ct_1 is None:
            ct_1 = torch.zeros((self.hidden_dim, embeds.shape[1])).to(device)
            ht_1 = torch.zeros((self.hidden_dim, embeds.shape[1])).to(device)

        for t, embed in enumerate(embeds):
            XH = torch.cat((embed.T, ht_1), dim=0)
            gt = torch.tanh(Wgxh@XH + self.bg)
            it = torch.sigmoid(Wixh@XH + self.bi)
            ft = torch.sigmoid(Wfxh@XH + self.bf)
            ot = torch.sigmoid(Woxh@XH + self.bo)
            ct = gt*it + ct_1*ft
            ht = torch.tanh(ct) * ot

            if out is None:
                out = ht.T.unsqueeze(0)
            else:
                out = torch.cat((out, ht.T.unsqueeze(0)), dim=0)
            ct_1 = ct
            ht_1 = ht

        return out, ht_1, ct_1
        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """
    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.vocab_size = args.vocabulary_size
        self.input_size = args.embedding_size
        self.hidden_dim = args.lstm_hidden_dim

        self.embedder = nn.Embedding(self.vocab_size, self.input_size)
        self.lstm = LSTM(self.hidden_dim, self.input_size)
        # Define output parameters.
        self.Wph = nn.Parameter(torch.empty((self.vocab_size, self.hidden_dim)))
        nn.init.uniform_(self.Wph, -1 / math.sqrt(self.hidden_dim),
                                               1 / math.sqrt(self.hidden_dim))
        self.bp = nn.Parameter(torch.empty((self.vocab_size, 1)))
        nn.init.uniform_(self.bp, -1 / math.sqrt(self.hidden_dim),
                                             1 / math.sqrt(self.hidden_dim))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.embedder(x)
        lstm_out = self.lstm(x)[0]
        T = lstm_out.shape[0]
        lin_out = (torch.matmul(self.Wph.repeat(T, 1, 1), lstm_out.transpose(1,2)) \
                   + self.bp).transpose(1, 2)
        outputs = nn.Softmax(dim=-1)(lin_out)
        return outputs
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=0.):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        device = next(self.parameters()).device
        init_chars = torch.randperm(self.vocab_size)[:batch_size].reshape(1, batch_size, 1)
        gen = torch.cat((
            init_chars,
            torch.zeros((sample_length - 1, batch_size, 1))
        )).type(torch.LongTensor).to(device)
        ht_1 = None
        ct_1 = None

        for t in range(sample_length - 1):
            embeds = self.embedder(gen[t])
            lstm_out, ht_1, ct_1 = self.lstm(embeds.view(1, batch_size, -1), ht_1, ct_1)
            lin_out = (torch.matmul(self.Wph.repeat(1, 1, 1), lstm_out.transpose(1,2)) \
                       + self.bp).transpose(1, 2)
            if temperature == 0:
                predictions = torch.argmax(nn.Softmax(dim=-1)(lin_out),
                                        dim=-1, keepdims=True)
            else:
                dist = nn.Softmax(dim=-1)(lin_out / temperature).squeeze()
                predictions = torch.multinomial(dist, 1)
            gen[t + 1] = predictions

        return gen

        #######################
        # END OF YOUR CODE    #
        #######################

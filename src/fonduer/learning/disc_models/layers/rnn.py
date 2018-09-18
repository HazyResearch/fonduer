"""
A recurrent neural network module.
"""

import torch
import torch.nn as nn


class RNN(nn.Module):
    """A recurrent neural network layer.

    :param num_classes: Number of classes.
    :type num_classes: int
    :param num_tokens: Size of embeddings.
    :type num_tokens: int
    :param emb_size: Dimension of embeddings.
    :type emb_size: int
    :param lstm_hidden: Size of LSTM hidden layer size.
    :type lstm_hidden: int
    :param num_layers: Number of recurrent layers.
    :type num_layers: int
    :param dropout: Dropout parameter of LSTM.
    :type dropout: float
    :param attention: Use attention or not.
    :type attention: bool
    :param bidirectional: Use bidirectional LSTM or not.
    :type bidirectional: bool
    :param use_cuda: Use use_cuda or not.
    :type use_cuda: bool
    """

    def __init__(
        self,
        num_classes,
        num_tokens,
        emb_size,
        lstm_hidden,
        num_layers=1,
        dropout=0.0,
        attention=True,
        bidirectional=True,
        use_cuda=False,
    ):

        super(RNN, self).__init__()

        self.num_tokens = num_tokens
        self.emb_size = emb_size
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        # Number of dimensions of output.  means no final linear layer.
        self.num_classes = num_classes

        if self.num_classes > 0:
            self.final_linear = True
        else:
            self.final_linear = False

        self.drop = nn.Dropout(dropout)
        self.lookup = nn.Embedding(num_tokens, emb_size, padding_idx=0)

        b = 2 if self.bidirectional else 1

        self.word_lstm = nn.LSTM(
            self.emb_size,
            self.lstm_hidden,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        if attention:
            self.attn_linear_w_1 = nn.Linear(
                b * lstm_hidden, b * lstm_hidden, bias=True
            )
            self.attn_linear_w_2 = nn.Linear(b * lstm_hidden, 1, bias=False)

        if self.final_linear:
            self.linear = nn.Linear(b * lstm_hidden, num_classes)

    def forward(self, x, x_mask, state_word):
        """Forward function.

        :param x: Input sequence tensor.
        :type x: torch.Tensor of shape (batch_size * length)
        :param x_mask: Use use_cuda or not.
        :type x_mask: torch.Tensor of shape (batch_size * length)
        :param state_word: Initial state of LSTM.
        :type state_word: torch.Tensor (see init_hidden() for more information)
        :return: Output of LSTM layer, either after mean pooling or attention.
        :rtype: torch.Tensor with shape (batch_size, num_directions * hidden_size)
            if num_classes > 0 otherwise with shape (batch_size, num_classes)
        """

        x_emb = self.drop(self.lookup(x))
        output_word, state_word = self.word_lstm(x_emb, state_word)
        output_word = self.drop(output_word)
        if self.attention:
            """
            An attention layer where the attention weight is
            a = T' . tanh(Wx + b)
            where x is the input, b is the bias.
            """
            word_squish = torch.tanh(self.attn_linear_w_1(output_word))
            word_attn = self.attn_linear_w_2(word_squish)
            word_attn.data.masked_fill_(x_mask.data.unsqueeze(dim=2), float("-inf"))
            word_attn_norm = torch.sigmoid(word_attn.squeeze(2))
            word_attn_vectors = torch.bmm(
                output_word.transpose(1, 2), word_attn_norm.unsqueeze(2)
            ).squeeze(2)
            output = (
                self.linear(word_attn_vectors)
                if self.final_linear
                else word_attn_vectors
            )
        else:
            """
            Mean pooling
            """
            x_lens = x_mask.data.eq(0).long().sum(dim=1)
            if self.use_cuda:
                weights = torch.ones(x.size()).cuda() / x_lens.unsqueeze(1).float()
            else:
                weights = torch.ones(x.size()) / x_lens.unsqueeze(1).float()
            weights.data.masked_fill_(x_mask.data.unsqueeze(dim=2), 0.0)
            word_vectors = torch.bmm(
                output_word.transpose(1, 2), weights.unsqueeze(2)
            ).squeeze(2)
            output = self.linear(word_vectors) if self.final_linear else word_vectors

        return output

    def init_hidden(self, batch_size):
        """Initiate the initial state.

        :param batch_size: batch size.
        :type batch_size: int
        :return: Initial state of LSTM
        :rtype: pair of torch.Tensors of shape (num_layers * num_directions,
            batch_size, hidden_size)
        """

        b = 2 if self.bidirectional else 1
        if self.use_cuda:
            return (
                torch.zeros(self.num_layers * b, batch_size, self.lstm_hidden).cuda(),
                torch.zeros(self.num_layers * b, batch_size, self.lstm_hidden).cuda(),
            )
        else:
            return (
                torch.zeros(self.num_layers * b, batch_size, self.lstm_hidden),
                torch.zeros(self.num_layers * b, batch_size, self.lstm_hidden),
            )

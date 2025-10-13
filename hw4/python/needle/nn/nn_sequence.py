"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (ops.exp(-x) + 1) ** -1
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device=device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if nonlinearity == "tanh":
          self.nonlinearity_fn = ops.tanh
        else:
          self.nonlinearity_fn = ops.relu
        low = - (1 / math.sqrt(hidden_size))
        high = -low
        self.W_ih = init.rand(*(input_size,hidden_size), low=low, high=high, device=device, dtype=dtype, requires_grad=True)
        self.W_hh = init.rand(*(hidden_size,hidden_size), low=low, high=high, device=device, dtype=dtype, requires_grad=True)
        if bias:
          self.bias_ih = init.rand((hidden_size), low=low,high=high, device=device, dtype=dtype, requires_grad=True)
          self.bias_hh = init.rand((hidden_size), low=low,high=high, device=device, dtype=dtype, requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        res = X @ self.W_ih
        if h:
            res += h @ self.W_hh
        if self.bias:
            res += (self.bias_ih + self.bias_hh).reshape((1,self.hidden_size)).broadcast_to(res.shape)
        return self.nonlinearity_fn(res)

        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers=num_layers
        self.rnn_cells = [RNNCell(input_size=input_size if k == 0 else hidden_size,
                                  hidden_size=hidden_size,
                                  bias=bias,
                                  nonlinearity=nonlinearity,
                                  device=device,
                                  dtype=dtype
                                ) for k in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h_seq = list(h0.split(0)) if h0 is not None else [None] * self.num_layers

        X_Split = list(X.split(0))
        out = []
        for seq_idx in range(X.shape[0]):
          for layer_idx in range(self.num_layers):
            x = X_Split[seq_idx] if layer_idx == 0 else h_seq[layer_idx - 1]
            h_seq[layer_idx] = self.rnn_cells[layer_idx](x,h_seq[layer_idx])
          out.append(h_seq[-1])
        return ops.stack(out,0) , ops.stack(h_seq,0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.bias = bias
        self.device = device
        self.sigmod = Sigmoid()
        self.hidden_size=hidden_size
        low = - (1 / math.sqrt(hidden_size))
        high = -low
        self.W_ih = init.rand(*(input_size,4*hidden_size), low=low, high=high, device=device, dtype=dtype, requires_grad=True)
        self.W_hh = init.rand(*(hidden_size,4*hidden_size), low=low, high=high, device=device, dtype=dtype, requires_grad=True)
        if bias:
          self.bias_ih = init.rand((4*hidden_size), low=low,high=high, device=device, dtype=dtype, requires_grad=True)
          self.bias_hh = init.rand((4*hidden_size), low=low,high=high, device=device, dtype=dtype, requires_grad=True)
        
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        res = X @ self.W_ih
        if h:
          h0,c0 = h
          res += h0 @ self.W_hh
        if self.bias:
          res += (self.bias_ih + self.bias_hh).reshape((1,4*self.hidden_size)).broadcast_to(res.shape)
        res = list(res.split(1))
        i = self.sigmod(ops.stack(res[:self.hidden_size],1))
        f = self.sigmod(ops.stack(res[self.hidden_size:2*self.hidden_size],1))
        g = ops.tanh(ops.stack(res[2*self.hidden_size:3*self.hidden_size],1))
        o = self.sigmod(ops.stack(res[3*self.hidden_size:],1))
        
        c = i * g
        if h:
          if c0.shape != f.shape:
            breakpoint()
          c += f * c0
        h = o * ops.tanh(c)
        return h , c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.bias = bias
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.device=device
        self.lstm_cells = [LSTMCell(input_size=input_size if i == 0 else hidden_size,
                          hidden_size=hidden_size,
                          bias=bias,
                          device=device,
                          dtype=dtype)
                          for i in range(self.num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
          h_seq = [None] * self.num_layers
          c_seq = [None] * self.num_layers
        else:
          h_seq = list(h[0].split(0))
          c_seq = list(h[1].split(0))
        X_split = X.split(0)
        out = []
        for seq_idx in range(X.shape[0]):
          for layer_idx in range(self.num_layers):
            x = X_split[seq_idx] if layer_idx == 0 else h_seq[layer_idx - 1]
            h = (h_seq[layer_idx], c_seq[layer_idx]) if h_seq[layer_idx] is not None else None
            h_seq[layer_idx], c_seq[layer_idx] = self.lstm_cells[layer_idx](x,h)
          out.append(h_seq[-1])
        return ops.stack(out,0),(ops.stack(h_seq,0), ops.stack(c_seq,0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        one_hot = init.one_hot(self.num_embeddings, x.reshape((x.shape[0]*x.shape[1], )), device=self.device)
        return  (one_hot @ self.weight).reshape((x.shape[0], x.shape[1], self.embedding_dim))
        ### END YOUR SOLUTION
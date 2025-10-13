import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(ndl.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
      super().__init__()
      self.models = nn.Sequential(
        nn.Conv(in_channels, out_channels, kernel_size, stride, bias, device, dtype),
        nn.BatchNorm2d(out_channels,device=device),
        nn.ReLU()
      )
  def forward(self, x: ndl.Tensor):
    return self.models(x)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        ### BEGIN YOUR SOLUTION ###
        self.conv1 = ConvBN(3,16,7,4,device=device)
        self.conv2 = ConvBN(16,32,3,2,device=device)
        self.conv3 = ConvBN(32,32,3,1,device=device)
        self.conv4 = ConvBN(32,32,3,1,device=device)
        self.conv5 = ConvBN(32,64,3,2,device=device)
        self.conv6 = ConvBN(64,128,3,2,device=device)
        self.conv7 = ConvBN(128,128,3,1,device=device)
        self.conv8 = ConvBN(128,128,3,1,device=device)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,128,device=device)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128,10,device=device)
        
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        # breakpoint()
        conv2 = self.conv2(self.conv1(ndl.Tensor(x, device=self.device)))
        conv4 = self.conv4(self.conv3(conv2)) + conv2
        conv6 = self.conv6(self.conv5(conv4))
        conv8 = self.conv8(self.conv7(conv6)) + conv6


        return self.linear2(self.relu(self.linear1(self.flatten(conv8))))


        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device
        self.seq_model_name = seq_model
        self.num_layers = num_layers
        seq_model_class = nn.RNN if seq_model == 'rnn' else nn.LSTM
        self.embedding = nn.Embedding(output_size, embedding_size, device=device)
        self.seq_model = seq_model_class(embedding_size,hidden_size,num_layers,device=device)
        self.linear = nn.Linear(hidden_size, output_size, device=device)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        
        if h is None:
          if self.seq_model_name == 'rnn':
            h = ndl.init.zeros(self.num_layers,x.shape[1],self.hidden_size,device=self.device)
          else:
            h = (ndl.init.zeros(self.num_layers,x.shape[1],self.hidden_size,device=self.device),) * 2
        
        x = self.embedding(x)
        out,h = self.seq_model(x,h)
        return self.linear(out.reshape((x.shape[0]*x.shape[1],self.hidden_size))),h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)

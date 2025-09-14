import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(dim, hidden_dim)
    norm1 = norm(hidden_dim)
    relu = nn.ReLU()
    dropout = nn.Dropout(drop_prob)
    linear2 = nn.Linear(hidden_dim, dim)
    norm2 = norm(dim)


    tmp_model = nn.Sequential(*[linear1,norm1,relu,dropout,linear2,norm2])
    res = nn.Residual(tmp_model)
    relu_out = nn.ReLU()
    ### END YOUR SOLUTION

    model = nn.Sequential(*[res,relu_out])
    return model


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    path = nn.Sequential(
      nn.Linear(dim,hidden_dim),
      nn.ReLU(),
      *[ResidualBlock(hidden_dim,hidden_dim//2,norm,drop_prob) for _ in range(num_blocks)],
      nn.Linear(hidden_dim,num_classes)
    )
    return path
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    hit,loss_sum,total = 0,0,0
    iteration = 0

    if opt is not None:
      model.train()
      for X_batch,y_batch in dataloader:
        iteration += 1
        opt.reset_grad()
        out = model(X_batch)
        loss = loss_func(out,y_batch)
        loss.backward()
        opt.step()
        hit += (out.numpy().argmax(1) == y_batch.numpy()).sum()
        loss_sum += loss.numpy()
        total += y_batch.shape[0]
    else:
      model.eval()
      for X_batch,y_batch in dataloader:
        iteration += 1
        out = model(X_batch)
        loss = loss_func(out,y_batch)
        hit += (out.numpy().argmax(1) == y_batch.numpy() ).sum()
        loss_sum += loss.numpy()
        total += y_batch.shape[0]

    return 1 - hit / total,loss_sum / iteration
    ### END YOUR SOLUTION



def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz",
        )
    test_dataset = ndl.data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz",
        data_dir + "/t10k-labels-idx1-ubyte.gz",
        )
    train_dataloader = ndl.data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        )
    test_dataloader = ndl.data.DataLoader(
        test_dataset,
        batch_size,
        )
    
    model = MLPResNet(784, hidden_dim)
    if optimizer is not None:
        opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        for e in range(epochs):
            train_acc, train_loss = epoch(train_dataloader, model, opt)
            if e == epochs - 1:
                test_acc, test_loss = epoch(test_dataloader, model)
        return (train_acc, train_loss, test_acc, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")

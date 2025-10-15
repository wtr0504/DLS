"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
          if self.u.get(w) is None:
            self.u[w] = ndl.zeros_like(w)
          grad = w.grad.detach() + self.weight_decay * w.detach()
          self.u[w].data = self.momentum * self.u[w].data + (1 - self.momentum) * grad.data
          w.data = w.data - self.lr * self.u[w].data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [ndl.zeros_like(x) for x in self.params]
        self.v = [ndl.zeros_like(x) for x in self.params]

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i,w in enumerate(self.params):
          
          grad = w.grad.detach() + self.weight_decay * w.detach()
          self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * grad.data
          self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (grad.data ** 2)
          u = self.m[i] / (1 - self.beta1 ** self.t)
          v = self.v[i] / (1 - self.beta2 ** self.t)
          w.data = w.data - self.lr * (u.data / (v.data ** 0.5 + self.eps))
        ### END YOUR SOLUTION
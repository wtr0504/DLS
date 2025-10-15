"""The module.
"""
from sys import modules
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from needle import cuda


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        kwargs = {"device" : device, "dtype" : dtype, "requires_grad":self.training}
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features,fan_out=out_features,**kwargs))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, **kwargs).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        tmp = X @ self.weight
        if self.bias:
          tmp += self.bias.broadcast_to(tmp.shape)
        return tmp
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        sz = 1
        for dim in X.shape[1:]:
          sz *= dim
        return X.reshape((X.shape[0],sz))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        tmp = x
        for m in self.modules:
          tmp = m(tmp)
        return tmp
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(logits.shape[1], y,device=logits.device,requires_grad=self.training)
        return (ops.logsumexp(logits, axes = (1,)) - (logits * one_hot).sum(axes = (1,))).sum() / y.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device,dtype=dtype,requires_grad=self.training))
        self.bias = Parameter(init.zeros(self.dim,device=device,dtype=dtype,requires_grad=self.training))
        self.running_mean = init.zeros(self.dim,device=device,dtype=dtype,requires_grad=self.training)
        self.running_var = init.ones(self.dim,device=device,dtype=dtype,requires_grad=self.training)

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if not self.training:
          return self.weight.broadcast_to(x.shape) * (x - self.running_mean.broadcast_to(x.shape)) / ((self.running_var + self.eps) ** 0.5).broadcast_to(x.shape) + self.bias.broadcast_to(x.shape) 
        e = x.sum(axes=(0,)) / x.shape[0]
        diff = x - e.broadcast_to(x.shape)
        var = (diff ** 2).sum(axes=(0,)) / x.shape[0]
        denominator = (var + self.eps) ** 0.5
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * e
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        return self.weight.broadcast_to(x.shape) * diff / denominator.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
  def __init__(self, *args,**kwargs):
    super().__init__(*args,**kwargs)
  
  def forward(self, x: Tensor) -> Tensor:
    x_ = x.permute((0,2,3,1)).reshape((x.shape[0] * x.shape[2] * x.shape[3], x.shape[1]))
    out = super().forward(x_).reshape(((x.shape[0], x.shape[2], x.shape[3], x.shape[1])))
    return out.permute((0,3,1,2))



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=self.training))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        e_x = x.sum(axes = (1,)) / self.dim
        temp_shape = (x.shape[0], 1)
        diff = x - e_x.reshape(temp_shape).broadcast_to(x.shape)
        var = (diff ** 2).sum(axes = (1,)) / self.dim
        denominator = (var + self.eps)**0.5
        return (self.weight.broadcast_to(x.shape) * diff / denominator.reshape(temp_shape).broadcast_to(x.shape) +
                self.bias.broadcast_to(x.shape))
        ### END YOUR SOLUTION



class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
          return x
        mask = init.randb(*x.shape,p=1 - self.p,device=x.device)
        return x * mask / (1 - self.p)

        
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
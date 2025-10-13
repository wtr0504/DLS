"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.pow(a,b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        base,exponent = node.inputs
        return out_grad * exponent * array_api.pow(base,exponent-1),out_grad * array_api.pow(base,exponent) * log(base)
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        base = node.inputs[0]
        return out_grad * (self.scalar) * (base ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs = node.inputs[0],node.inputs[1]
        return out_grad / rhs , (-1) * out_grad * (lhs / rhs ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dim = len(a.shape)
        if self.axes is None:
          self.axes = (dim - 2,dim - 1)
        axes = list(range(dim))

        axes[self.axes[0]] = self.axes[1]
        axes[self.axes[1]] = self.axes[0]

        return array_api.transpose(a,axes=axes)
        
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        axes = tuple([len(self.shape) - ind - 1 for ind, dim in enumerate(reversed(self.shape)) if
                      len(input_shape) - ind - 1 < 0 or dim != input_shape[-ind - 1]])
        return out_grad.sum(axes = axes).reshape(input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = self.axes,
        elif self.axes is None:
            axes = list(range(len(a.shape)))
        else:
            raise TypeError
        res = a
        for axis in sorted(axes, reverse=True):
            res=res.sum(axis)
        return res

    def gradient(self, out_grad, node):
        new_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(new_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError("Unsupported axes type, must be int, tuple or None!")
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)



def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs = node.inputs
        lgrad,rgrad = matmul(out_grad , transpose(rhs)) , matmul(transpose(lhs) , out_grad)
        
        if(len(lgrad.shape) > len(lhs.shape)):
          lgrad = lgrad.sum(tuple(range(len(lgrad.shape) - len(lhs.shape))))
        if((len(rgrad.shape) > len(rhs.shape))):
          rgrad = rgrad.sum(tuple(range(len(rgrad.shape) - len(rhs.shape))))
        return lgrad,rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return exp(node.inputs[0]) * out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # out = node.realize_cached_data()
        # out[out > 0] = 1
        # return out_grad * Tensor(out,device=out_grad.device)
        out = node.realize_cached_data()
        mask_data = (out > 0)
        if mask_data.shape != out_grad.shape:
          breakpoint()
        return out_grad * Tensor(mask_data, device=out_grad.device)
        # return out_grad * Tensor(node.inputs[0].realize_cached_data() > 0, device=out_grad.device)
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        out = 1 - out ** 2
        return out_grad * out
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, len(args))
        res = array_api.empty(new_shape, device=args[0].device)
        for i, arr in enumerate(args):
            slices = tuple(i if ind == self.axis else slice(None) for ind, dim in enumerate(new_shape))
            res[slices] = arr
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(0, s) for s in A.shape]
        splits = []
        for i in range(n):
          slices[self.axis] = slice(i, i + 1)
          splits.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(splits)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad,self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
          new_shape[axis] = new_shape[axis] * (self.dilation + 1)
        out = array_api.full(new_shape,0,dtype=a.dtype,device=a.device)
        slices = tuple(slice(None,None,self.dilation + 1) if idx in self.axes else slice(None) for idx in range(len(a.shape)))
        out[slices] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
          new_shape[axis] = new_shape[axis] / (self.dilation + 1)
        slices = tuple(slice(None,None,self.dilation + 1) if idx in self.axes else slice(None) for idx in range(len(a.shape)))
        return a[slices]

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        S = self.stride

        A_padded = A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        new_shape = N, (A_padded.shape[1] - K + S) // S, (A_padded.shape[2] - K + S) // S, K , K, C_in,
        new_strides =  A_padded.strides[0],  S * C_in * A_padded.shape[2], S * C_in, C_in * A_padded.shape[2] , C_in, 1,
        A = A_padded.as_strided(new_shape,new_strides)
        A = A.compact().reshape((N * new_shape[1] * new_shape[2],K * K * C_in))
        B = B.compact().reshape((K * K * C_in,C_out))
        return (A @ B).compact().reshape((N, new_shape[1], new_shape[2], C_out))

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        out_grad_as_A = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)
        B_ = flip(B, (0,1)).transpose()
        padding = (A.shape[2] - out_grad_as_A.shape[2] + B_.shape[0] - 1) // 2
        A_grad = conv(out_grad_as_A, B_, padding= padding)

        A_ = A.transpose((0,3))
        out_grad_as_B = dilate(out_grad.transpose((0, 1)).transpose((1,2)), axes=(0, 1), dilation=self.stride - 1)
        B_grad = conv(A_, out_grad_as_B, padding= self.padding).transpose((0, 1)).transpose((1,2))
        return A_grad, B_grad

        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)

class Permute(TensorOp):
  def __init__(self,axes):
    self.axes = axes

  def compute(self, a):
    return array_api.permute(a,self.axes)
  
  def gradient(self, out_grad,node):
    inv_axes = [0] * len(self.axes)
    for i, a in enumerate(self.axes):
        inv_axes[a] = i
    # 应该用逆置换来还原梯度
    return out_grad.permute(inv_axes)

def permute(a,axes):
  return Permute(axes)(a)


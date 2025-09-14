from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api
import needle

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z,axis=1,keepdims=True)
        exp_max_z = Z - max_z
        sum_exp_max_minus_z = array_api.sum(array_api.exp(exp_max_z),axis=1,keepdims=True)
        return exp_max_z - array_api.log(sum_exp_max_minus_z)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        y = node.realize_cached_data().copy()
        z = node.inputs[0]
        tiled_shape = (z.shape[0], z.shape[1], z.shape[1])
        tiled_softmax = Tensor(array_api.exp(y).reshape(z.shape[0], z.shape[1], 1)).broadcast_to(tiled_shape)
        tiled_eyes = Tensor(array_api.identity(z.shape[1])).broadcast_to(tiled_shape)
        tiled_out_grad = out_grad.reshape((z.shape[0], 1, z.shape[1])).broadcast_to(tiled_shape)
        return (tiled_out_grad * (tiled_eyes - tiled_softmax)).sum(2).reshape(z.shape)


        
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z,self.axes)
        max_z_ = array_api.max(Z,self.axes,keepdims=True)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_),axis=self.axes)) + max_z
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        axes = self.axes or range(len(z.shape))
        tmp_shape = [1 if i in axes else dim for i,dim in enumerate(z.shape)]
        max_z = array_api.max(z.cached_data,self.axes,keepdims=True)
        exp_z = needle.exp(z - Tensor(max_z))
        return exp_z * (out_grad / exp_z.sum(axes=self.axes)).reshape(tmp_shape).broadcast_to(z.shape)
      

        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


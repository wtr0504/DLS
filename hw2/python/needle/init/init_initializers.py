import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * pow(6 / (fan_in + fan_out),0.5)
    shape = (fan_in,fan_out)
    return rand(*shape,low=-a, high=a,**kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    shape = (math.ceil(fan_in),math.ceil(fan_out))
    return randn(*shape,mean=0.0,std=std,**kwargs)
    # std = gain * math.sqrt(2 / (fan_in + fan_out))
    # return randn(fan_in, fan_out, std=std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = 2 ** 0.5
    bound = gain * ((3 / fan_in) ** 0.5)
    shape = (fan_in,fan_out)
    return rand(*shape,low= -bound,high=bound,**kwargs)
    
    ### END YOUR SOLUTION



def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = 2 ** 0.5
    std = gain / (fan_in ** 0.5)
    shape = (math.ceil(fan_in),math.ceil(fan_out))
    return randn(*shape,mean=0.0,std=std,**kwargs)
    ### END YOUR SOLUTION


import torch.nn as nn


def set_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'softmax':
        return nn.Softmax()
    else:
        raise ValueError(f'Activation {activation} not supported')

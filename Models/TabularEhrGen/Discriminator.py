import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, activation='relu'):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

        dim = input_dim
        seq = []
        for item in list(hidden_dims):
            seq += [
                SN(nn.Linear(dim, item)),
                nn.LeakyReLU(0.2),
            ]
            dim = item

        seq.append(nn.Linear(dim, output_dim))
        self.layers = nn.Sequential(*seq)

    def forward(self, x):
        out = self.layers(x)

        return out

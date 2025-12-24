import torch
import torch.nn as nn

from Utils.model import set_activation
from Models.AutoEncoder.ResidualBlock import ResidualBlock


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 compress_dims: list,
                 embedding_dim: int,
                 seq_len: int=None,
                 activation: str='relu'):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.compress_dims = compress_dims
        self.embedding_dim = embedding_dim
        self.activation = activation

        if seq_len is None:
            dim = input_dim
        else:
            dim = input_dim * seq_len
        seq = []
        for item in compress_dims:
            seq += [ResidualBlock(dim, item, activation), torch.nn.Dropout(0.2)]
            dim = item
        seq += [
            nn.Linear(dim, embedding_dim),
        ]
        self.layers = nn.Sequential(*seq)

    def forward(self, x):
        out = self.layers(x)

        return out


if __name__ == '__main__':
    encoder = Encoder(input_dim=32, compress_dims=[16, 8], embedding_dim=4)
    x = torch.randn(128, 32)
    out = encoder(x)
    breakpoint()
from typing import List

import torch
import torch.nn as nn

from Utils.model import set_activation
from Models.AutoEncoder.ResidualBlock import ResidualBlock


class MultiHeadDecoder(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 decompress_dims: list,
                 output_dims: List[int],
                 seq_len: int=None,
                 activation: str='relu'):
        super(MultiHeadDecoder, self).__init__()

        self.output_dim = output_dims
        self.decompress_dims = decompress_dims
        self.embedding_dim = embedding_dim
        self.activation = activation

        dim = embedding_dim
        seq = []
        for item in decompress_dims:
            seq += [ResidualBlock(dim, item, activation)]
            dim = item
        self.layers = nn.Sequential(*seq)

        if seq_len is None:
            self.multi_head = nn.ModuleList(
                [nn.Linear(dim, head_dim) for head_dim in output_dims]
            )
        else:
            self.multi_head = nn.ModuleList(
                [nn.Linear(dim, seq_len*head_dim) for head_dim in output_dims]
            )

    def forward(self, x):
        out = self.layers(x)
        out = [head(out) for head in self.multi_head]

        return out


if __name__ == '__main__':
    decoder = MultiHeadDecoder(embedding_dim=4, decompress_dims=[8, 16],
                               output_dims=[4, 8, 8, 4, 4, 4])
    x = torch.randn(128, 4)
    out = decoder(x)
    breakpoint()
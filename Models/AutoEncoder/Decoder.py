import torch
import torch.nn as nn

from Utils.model import set_activation
# from Models.AutoEncoder.ResidualBlock import ResidualBlock


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu'):
        super(ResidualBlock, self).__init__()

        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.fc = nn.Linear(input_dim, output_dim)
        # self.bn = nn.BatchNorm1d(output_dim)
        self.activation = set_activation(activation)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc(x)
        # out = self.bn(out)
        out = self.activation(out)

        return out + residual


class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 decompress_dims: list,
                 output_dim: int,
                 activation: str='relu'):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.decompress_dims = decompress_dims
        self.embedding_dim = embedding_dim
        self.activation = activation

        dim = embedding_dim
        seq = []
        for item in decompress_dims:
            seq += [ResidualBlock(dim, item, activation)]
            dim = item
        seq.append(nn.Linear(dim, output_dim))
        self.layers = nn.Sequential(*seq)

    def forward(self, x):
        out = self.layers(x)

        return out


if __name__ == '__main__':
    decoder = Decoder(embedding_dim=4, decompress_dims=[8, 16], output_dim=32)
    x = torch.randn(128, 4)
    out = decoder(x)
    breakpoint()
import torch
import torch.nn as nn

from Utils.model import set_activation


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu'):
        super(ResidualBlock, self).__init__()

        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.fc = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = set_activation(activation)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc(x)
        out = self.norm(out)
        out = self.activation(out)

        return out + residual

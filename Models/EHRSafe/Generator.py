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


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

        dim = input_dim
        seq = []
        for item in list(hidden_dims):
            seq += [ResidualBlock(dim, item, activation)]
            dim = item
        seq += [nn.Linear(dim, output_dim)]
        self.layers = nn.Sequential(*seq)

    def forward(self, x):
        out = self.layers(x)

        return out

    def generate_random_noise_vector(self,
                                     batch_size: int,
                                     dim: int,
                                     device: torch.device):
        mean = torch.zeros(batch_size, dim, device=device)
        std = torch.ones(batch_size, dim, device=device)
        z = torch.normal(mean=mean, std=std)
        return z

    def sample(self, n_samples, device='cpu'):
        self.eval()

        with torch.no_grad():
            z_dim = self.input_dim
            z = self.generate_random_noise_vector(n_samples, z_dim, device=device)
            samples = self.forward(z)

        return samples


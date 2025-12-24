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
    def __init__(self, latent_dim, input_dim, hidden_dims, output_dim, z_s_dim, z_t_dim, activation='relu'):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        self.z_s_dim = z_s_dim
        self.z_t_dim = z_t_dim

        dim = input_dim
        seq = []
        for item in list(hidden_dims):
            seq += [ResidualBlock(dim, item, activation)]
            dim = item
        self.shared_layers = nn.Sequential(*seq)

        self.z_s_head = nn.Linear(dim, z_s_dim)
        self.z_t_head = nn.Linear(dim, z_t_dim)

    def forward(self, x):
        shared = self.shared_layers(x)
        z_s = self.z_s_head(shared)
        z_t = self.z_t_head(shared)
        return torch.cat([z_s, z_t], dim=1)

    def generate_random_noise_vector(self,
                                     batch_size: int,
                                     dim: int,
                                     device: torch.device):
        mean = torch.zeros(batch_size, dim, device=device)
        std = torch.ones(batch_size, dim, device=device)
        z = torch.normal(mean=mean, std=std)
        return z

    def sample(self, n_samples, condition=None, device='cpu'):
        self.eval()

        with torch.no_grad():
            z_dim = self.latent_dim
            z = self.generate_random_noise_vector(n_samples, z_dim, device=device)

            if condition is not None:
                z = torch.cat((z, condition), dim=1)

            samples = self.forward(z)
        return samples


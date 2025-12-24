import torch
import torch.nn as nn

from Utils.model import set_activation
from Models.EHR_Safe.Generator import Generator
from Models.EHR_Safe.Discriminator import Discriminator
from Utils.namespace import _namespace_to_dict


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator

    def calculate_gradient_penalty(self,
                                   real: torch.Tensor,
                                   fake: torch.Tensor,
                                   device: torch.device,
                                   lambda_gp: int=10):
        batch_size, _ = real.size()

        epsilon = torch.rand(batch_size, 1, device=device).repeat(1, real.size(1))
        interpolates = epsilon * real + ((1 - epsilon) * fake)
        interpolates = interpolates.requires_grad_(True)

        disc_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_gp

        return gradient_penalty

    def forward(self, x):
        out = dict()

        batch_size, embedding_dim = x.size()
        device = x.device

        # generate random noise vector
        z_dim = self.generator.input_dim
        z = self.generator.generate_random_noise_vector(batch_size, z_dim, device)
        fake = self.generator(z)

        disc_fake = self.discriminator(fake)
        disc_real = self.discriminator(x)

        out['fake'] = fake
        out['disc_fake'] = disc_fake
        out['disc_real'] = disc_real

        return out


def build_model(model_config, device=torch.device('cpu')):
    print('Building GAN model')

    generator = Generator(**_namespace_to_dict(model_config.generator))
    discriminator = Discriminator(**_namespace_to_dict(model_config.discriminator))

    return GAN(generator=generator, discriminator=discriminator).to(device)

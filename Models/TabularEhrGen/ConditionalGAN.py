import torch
import torch.nn as nn

from Utils.model import set_activation
from Models.TabularEhrGen.Generator import Generator
from Models.TabularEhrGen.Discriminator import Discriminator
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
                                   lambda_gp: int=10,
                                   condition=None):
        batch_size, _ = real.size()

        epsilon = torch.rand(batch_size, 1, device=device)
        interpolates = epsilon * real + ((1 - epsilon) * fake)
        interpolates = interpolates.requires_grad_(True)

        if condition is not None:
            c = condition.detach()
            d_in = torch.cat([interpolates, c], dim=1)
        else:
            d_in = interpolates

        disc_interpolates = self.discriminator(d_in)

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

    def forward(self, x, condition=None):
        out = dict()

        batch_size, embedding_dim = x.size()
        device = x.device

        # generate random noise vector
        z_dim = self.generator.latent_dim
        z = self.generator.generate_random_noise_vector(batch_size, z_dim, device)
        if condition is not None:
            z = torch.cat((z, condition), dim=1)
        fake = self.generator(z)

        if condition is not None:
            disc_fake = self.discriminator(torch.cat([fake, condition], dim=1))
            disc_real = self.discriminator(torch.cat([x, condition], dim=1))
        else:
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

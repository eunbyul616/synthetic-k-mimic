import torch
import torch.nn as nn

from Models.AutoEncoder import Encoder, Decoder


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        rep = self.encoder(x)
        x_hat = self.decoder(rep)

        return rep, x_hat


def build_model(model_config, device=torch.device('cpu')):
    print("Building Autoencoder model")
    encoder = Encoder(**model_config['encoder'])
    decoder = Decoder(**model_config['decoder'])

    return Autoencoder(encoder=encoder, decoder=decoder).to(device)


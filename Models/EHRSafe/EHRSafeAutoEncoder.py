import torch
import torch.nn as nn

from Models.AutoEncoder import Encoder, Decoder
from Utils.namespace import _namespace_to_dict


class EHRSafeAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EHRSafeAutoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        rep = self.encoder(x)
        x_hat = self.decoder(rep)

        return rep, x_hat


def build_model(model_config, device=torch.device('cpu')):
    print("Building EHR-Safe Autoencoder model")
    encoder = Encoder(**_namespace_to_dict(model_config.encoder))
    decoder = Decoder(**_namespace_to_dict(model_config.decoder))

    return EHRSafeAutoencoder(encoder=encoder, decoder=decoder).to(device)

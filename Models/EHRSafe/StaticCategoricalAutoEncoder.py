import torch
import torch.nn as nn

from Models.AutoEncoder import Encoder, MultiHeadDecoder
from Utils.namespace import _namespace_to_dict


class StaticCategoricalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(StaticCategoricalAutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        rep = self.encoder(x)
        x_hat = self.decoder(rep)

        return rep, x_hat


def build_model(model_config, device=torch.device('cpu')):
    print("Building Static Categorical Autoencoder model")
    encoder = Encoder(**_namespace_to_dict(model_config.encoder))
    decoder = MultiHeadDecoder(**_namespace_to_dict(model_config.decoder))

    return StaticCategoricalAutoEncoder(encoder=encoder, decoder=decoder).to(device)

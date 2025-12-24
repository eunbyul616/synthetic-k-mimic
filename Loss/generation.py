import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.model import set_activation


class WGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super(WGANDiscriminatorLoss, self).__init__()

    def forward(self, disc_fake, disc_real):
        loss = (torch.mean(disc_fake) - torch.mean(disc_real))

        return loss


class WGANGeneratorLoss(nn.Module):
    def __init__(self):
        super(WGANGeneratorLoss, self).__init__()

    def forward(self, disc_fake, disc_real):
        loss = -torch.mean(disc_fake)

        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, disc_fake, disc_real, mask=None, is_wrong_condition=False):
        if is_wrong_condition:
            bce_loss = self.criterion(disc_real, torch.zeros_like(disc_real))
            if mask is not None:
                bce_loss = bce_loss * mask.unsqueeze(dim=-1).repeat(1, 1, disc_real.size(-1))

        else:
            bce_loss = (self.criterion(disc_real, torch.ones_like(disc_real)) +
                        self.criterion(disc_fake, torch.zeros_like(disc_fake)))
            if mask is not None:
                bce_loss = bce_loss * mask.unsqueeze(dim=-1).repeat(1, 1, disc_real.size(-1))

        return bce_loss.mean()


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, disc_fake, disc_real, mask=None):
        bce_loss = self.criterion(disc_fake, torch.ones_like(disc_fake))

        if mask is not None:
            bce_loss = bce_loss * mask.unsqueeze(dim=-1).repeat(1, 1, disc_fake.size(-1))

        return bce_loss.mean()


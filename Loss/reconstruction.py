import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x_hat, x, mask=None):
        if mask is None:
            loss = self.criterion(x_hat, x)
        else:
            loss = self.criterion(x_hat, x)
            loss = loss * mask.view(loss.size())

        return loss.mean()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x_hat, x, mask=None):
        if mask is None:
            loss = self.criterion(x_hat, x)
        else:
            loss = self.criterion(x_hat, x)
            loss = loss * mask.view(loss.size())

        return loss.mean()

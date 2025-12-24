import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticEncoder(nn.Module):
    def __init__(self, num_numerical, categorical_card, latent_dim, hidden_dim, mask=False, activation='relu'):
        super().__init__()
        self.mask = mask
        if mask:
            input_dim = num_numerical + len(categorical_card[:-2]) + sum(categorical_card[-2:]) + (
                    num_numerical + len(categorical_card))
        else:
            input_dim = num_numerical + len(categorical_card[:-2]) + sum(categorical_card[-2:])

        self.layers = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_num, x_cat, x_bin, x_listwise, x_mask=None):
        B, _ = x_cat.shape

        if x_bin is None:
            x_bin = torch.empty((B, 0), device=x_cat.device)
        if x_listwise is None:
            x_listwise = torch.empty((B, 0), device=x_cat.device)

        if x_mask is not None and self.mask:
            x = torch.cat([x_num, x_cat, x_bin, x_listwise, x_mask], dim=1)
        else:
            x = torch.cat([x_num, x_cat, x_bin, x_listwise], dim=1)

        h = F.relu(self.layers(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class StaticDecoder(nn.Module):
    def __init__(self, num_numerical, categorical_card, latent_dim, hidden_dim, mask=False):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_num = nn.Linear(hidden_dim, num_numerical)
        self.fc_cat = nn.ModuleList([nn.Linear(hidden_dim, card) for card in categorical_card[:-2] if card > 1])
        self.fc_bin = nn.ModuleList([nn.Linear(hidden_dim, card) for card in categorical_card[:-2] if card == 1])
        self.fc_likewise = nn.ModuleList([nn.Linear(hidden_dim, card) for card in categorical_card[-2:]])
        self.fc_mask = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_numerical + len(categorical_card))]) if mask else None

    def gumbel_softmax_sample(self, logits, tau=1.0, hard=False):
        y = F.gumbel_softmax(logits, tau=tau, hard=hard)
        return y

    def binary_gumbel_sigmoid(self, logits, tau=1.0, hard=False):
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        y = torch.sigmoid((logits + gumbel_noise) / tau)

        if hard:
            y_hard = (y > 0.5).float()
            y = (y_hard - y).detach() + y  # straight-through
        return y

    def forward(self, z, use_gumbel=True, gumbel_tau=1.0, hard=False):
        h = F.relu(self.fc1(z))
        num_out = self.fc_num(h)
        cat_out_logits = []
        for fc in self.fc_cat:
            logits = fc(h)
            cat_out_logits.append(logits)

        bin_out_logits = [fc(h) for fc in self.fc_bin] if len(self.fc_bin) > 0 else []
        if (len(bin_out_logits) > 0) and use_gumbel:
            bin_out_logits = [self.binary_gumbel_sigmoid(logits, tau=gumbel_tau, hard=hard) for logits in
                              bin_out_logits]

        listwise_out_logits = []
        for fc in self.fc_likewise:
            logits = fc(h)
            if use_gumbel:
                gumbel_out = self.binary_gumbel_sigmoid(logits, tau=gumbel_tau, hard=hard)
                listwise_out_logits.append(gumbel_out)
            else:
                listwise_out_logits.append(logits)

        mask_out_logits = [fc(h) for fc in self.fc_mask] if self.fc_mask is not None else None
        if (mask_out_logits is not None) and use_gumbel:
            mask_out_logits = [
                self.binary_gumbel_sigmoid(logits, tau=gumbel_tau, hard=hard)
                for logits in mask_out_logits
            ]

        return num_out, cat_out_logits, bin_out_logits, listwise_out_logits, mask_out_logits


class StaticVAE(nn.Module):
    def __init__(self, latent_dim, num_numerical, categorical_card, hidden_dim, mask=False,
                 conditional=True, condition_classes=1):
        super().__init__()
        self.encoder = StaticEncoder(latent_dim=latent_dim, num_numerical=num_numerical,
                                     categorical_card=categorical_card, hidden_dim=hidden_dim, mask=mask)
        self.decoder = StaticDecoder(latent_dim=latent_dim, num_numerical=num_numerical,
                                     categorical_card=categorical_card, hidden_dim=hidden_dim, mask=mask)

        if conditional:
            self.condition_fc = nn.Linear(latent_dim, condition_classes)
        else:
            self.condition_fc = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat, x_bin=None, x_listwise=None, x_mask=None,
                use_gumbel=True, gumbel_tau=1.0, hard=False):
        mu, logvar = self.encoder(x_num, x_cat, x_bin, x_listwise, x_mask)
        z = self.reparameterize(mu, logvar)
        z = nn.functional.layer_norm(z, z.shape[1:])

        condition_hat = self.condition_fc(z) if self.condition_fc is not None else None

        x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits = self.decoder(z,
                                                                                               use_gumbel=use_gumbel,
                                                                                               gumbel_tau=gumbel_tau,
                                                                                               hard=hard)
        return (x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits,
                mu, logvar, z, condition_hat)


def build_model(model_cfg, device):
    model = StaticVAE(latent_dim=model_cfg.latent_dim,
                      num_numerical=model_cfg.num_numerical,
                      categorical_card=model_cfg.categorical_card,
                      hidden_dim=model_cfg.hidden_dim,
                      mask=model_cfg.mask,
                      conditional=model_cfg.conditional,
                      condition_classes=model_cfg.condition_classes)
    return model.to(device)

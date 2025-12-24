import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    def __init__(self, embedding_dim, categorical_card, num_numerical, latent_dim, hidden_dim,
                 bidirectional=False, mask=False, n_layers=1, z_s_dim=0):
        super().__init__()

        self.bidirectional = bidirectional
        self.mask = mask
        self.z_s_dim = z_s_dim

        self.binary_idx = [i for i, card in enumerate(categorical_card[:-1]) if card == 1]
        self.cat_idx = [i for i, card in enumerate(categorical_card[:-1]) if card > 1]
        self.cat_embs = nn.ModuleList([
            nn.Embedding(card + 1, embedding_dim, padding_idx=card)
            for i, card in enumerate(categorical_card[:-1]) if card > 1
        ])

        if mask:
            base = (num_numerical + embedding_dim * len(self.cat_idx) + len(self.binary_idx) + categorical_card[-1] +
                    (num_numerical + len(self.cat_idx)) + len(self.binary_idx) + 1)

        else:
            base = num_numerical + embedding_dim * len(self.cat_idx) + len(self.binary_idx) + categorical_card[-1]
        input_dim = base + self.z_s_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional, num_layers=n_layers)

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc_mu = nn.Linear(lstm_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(lstm_output_dim, latent_dim)

    def forward(self, x_num, x_cat, x_bin, x_listwise, x_mask=None, z_s=None):
        B, T, _ = x_cat.shape

        x_cat_emb = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.cat_embs)]
        x_cat_emb = torch.cat(x_cat_emb, dim=-1)

        if x_bin is None:
            x_bin = torch.empty((B, T, 0), device=x_cat.device)
        if x_listwise is None:
            x_listwise = torch.empty((B, T, 0), device=x_cat.device)

        if x_mask is not None and self.mask:
            x = torch.cat([x_num, x_cat_emb, x_bin, x_listwise, x_mask], dim=-1)
        else:
            x = torch.cat([x_num, x_cat_emb, x_bin, x_listwise], dim=-1)

        if self.z_s_dim > 0:
            assert z_s is not None, "[TemporalEncoder] z_s is required when z_s_dim > 0"
            z_s_seq = z_s.unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([x, z_s_seq], dim=-1)

        out, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]

        return self.fc_mu(h), self.fc_logvar(h), out


class TemporalDecoder(nn.Module):
    def __init__(self, latent_dim, num_numerical, categorical_card, hidden_dim, mask=False, seq_len=30,
                 z_s_dim=0):
        super().__init__()

        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.z_s_dim = z_s_dim

        self.fc1 = nn.Linear(latent_dim + self.z_s_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)

        self.fc_num = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_numerical)
        )
        self.fc_cat = nn.ModuleList([nn.Linear(hidden_dim, card) for card in categorical_card[:-1] if card > 1])
        self.fc_bin = nn.ModuleList([nn.Linear(hidden_dim, 1) for card in categorical_card[:-1] if card == 1])
        self.fc_likewise = nn.ModuleList([nn.Linear(hidden_dim, card) for card in categorical_card[-1:]])
        self.fc_mask = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_numerical + len(categorical_card))]) if mask else None

    def binary_gumbel_sigmoid(self, logits, tau=1.0, hard=False):
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        y = torch.sigmoid((logits + gumbel_noise) / tau)
        if hard:
            y_hard = (y > 0.5).float()
            y = (y_hard - y).detach() + y  # Straight-through
        return y

    def forward(self, z, seq_len, use_gumbel=True, gumbel_tau=1.0, hard=False, z_s=None):
        B = z.size(0)

        if self.z_s_dim > 0:
            assert z_s is not None, "[TemporalDecoder] z_s is required when z_s_dim > 0"
            z_cat = torch.cat([z, z_s], dim=-1)
        else:
            z_cat = z

        pos = torch.arange(seq_len, device=z.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        z_emb = F.relu(self.fc1(z_cat)).unsqueeze(1)
        h0 = z_emb + pos_emb

        out, _ = self.lstm(h0)

        num_out = self.fc_num(out)  # (B, T, num_numerical)
        cat_out_logits = [fc(out) for fc in self.fc_cat]  # List of (B, T, card_i)
        bin_out_logits = [fc(out) for fc in self.fc_bin] if len(self.fc_bin) > 0 else []
        if bin_out_logits is not None and use_gumbel:
            bin_out_logits = [
                self.binary_gumbel_sigmoid(logits, tau=gumbel_tau, hard=hard)
                for logits in bin_out_logits
            ]

        listwise_out_logits = []
        for fc in self.fc_likewise:
            logits = fc(out)
            if use_gumbel:
                gumbel_out = self.binary_gumbel_sigmoid(logits, tau=gumbel_tau, hard=hard)
                listwise_out_logits.append(gumbel_out)
            else:
                listwise_out_logits.append(logits)

        mask_out_logits = [fc(out) for fc in self.fc_mask] if self.fc_mask is not None else None
        if (mask_out_logits is not None) and use_gumbel:
            mask_out_logits = [
                self.binary_gumbel_sigmoid(logits, tau=gumbel_tau, hard=hard)
                for logits in mask_out_logits
            ]

        return num_out, cat_out_logits, bin_out_logits, listwise_out_logits, mask_out_logits


class TemporalVAE(nn.Module):
    def __init__(self, embedding_dim, categorical_card, num_numerical, latent_dim, hidden_dim,
                 mask=False, bidirectional=False, n_layers=1,
                 conditional=True, condition_classes=1, z_s_dim=0):
        super().__init__()
        self.encoder = TemporalEncoder(embedding_dim, categorical_card, num_numerical, latent_dim, hidden_dim,
                                       mask, bidirectional, n_layers, z_s_dim=z_s_dim)
        self.decoder = TemporalDecoder(latent_dim, num_numerical, categorical_card, hidden_dim, mask, z_s_dim=z_s_dim)

        if conditional:
            self.condition_fc = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(latent_dim // 2, condition_classes)
            )
        else:
            self.condition_fc = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat, x_bin=None, x_listwise=None, x_mask=None, use_gumbel=True, gumbel_tau=1.0,
                hard=False, z_s=None):
        mu, logvar, enc_hidden = self.encoder(x_num, x_cat, x_bin, x_listwise, x_mask, z_s=z_s)
        z = self.reparameterize(mu, logvar)
        z = nn.functional.layer_norm(z, z.shape[1:])

        condition_hat = self.condition_fc(z) if self.condition_fc is not None else None

        x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, mask_out_logits = self.decoder(z,
                                                                                                 x_num.size(1),
                                                                                                 use_gumbel=use_gumbel,
                                                                                                 gumbel_tau=gumbel_tau,
                                                                                                 hard=hard,
                                                                                                 z_s=z_s)

        return (x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, mask_out_logits,
                mu, logvar, z, condition_hat)


def build_model(model_cfg, device):
    model = TemporalVAE(
        embedding_dim=model_cfg.embedding_dim,
        categorical_card=model_cfg.categorical_card,
        num_numerical=model_cfg.num_numerical,
        latent_dim=model_cfg.latent_dim,
        hidden_dim=model_cfg.hidden_dim,
        mask=model_cfg.mask,
        bidirectional=model_cfg.bidirectional,
        n_layers=model_cfg.n_layers,
        conditional=model_cfg.conditional,
        condition_classes=model_cfg.condition_classes,
        z_s_dim=getattr(model_cfg, 'z_s_dim', 0)
    )
    return model.to(device)

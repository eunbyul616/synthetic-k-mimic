import os
from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from Trainer.train import Trainer
from Trainer.utils import *
from DataLoaders.loaderbase import get_dataloaders
from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset

from Utils.reproducibility import lock_seed
from Utils.train import set_loss_fn, set_scheduler, set_optimizer
from Utils.train import save_ckpt, initialize_weights
from Utils.namespace import set_cfg

from Models.TabularEhrGen.StaticVAE import build_model

from Visualization import *
from Evaluation.DistributionSimilarity.correlation_comparison import *

import config_manager

import torch
import torch.nn.functional as F


class StaticVAETrainer(Trainer):
    def __init__(self, config, rank, **kwargs):
        super(StaticVAETrainer, self).__init__(config=config, rank=rank, **kwargs)

        self.dataloaders = get_dataloaders(cfg=self.cfg,
                                           train_dataset=kwargs['train_dataset'],
                                           valid_dataset=kwargs['validation_dataset'],
                                           test_dataset=kwargs['test_dataset'],
                                           rank=self.rank,
                                           world_size=self.world_size,
                                           collate_fn=kwargs['collate_fn'],
                                           train_sampler=kwargs['train_sampler'],
                                           validation_sampler=kwargs['validation_sampler'])
        self.test_loss = {key: [] for key in self.loss_keys}

        self.transformer = self.dataloaders.train.dataset.static_transformer
        sc_cols = []
        sb_cols = []
        for c in self.dataloaders.train.dataset.sc_cols:
            info = [info for info in self.transformer._data_manipulation_info_list if info.column_name == c][0]
            dim = info.output_dimensions
            if dim == 1:
                sb_cols.append(c)
            else:
                sc_cols.append(c)

        self.sb_cols = sb_cols
        self.sc_cols = sc_cols
        self.sn_cols = self.dataloaders.train.dataset.sn_cols
        self.sl_cols = self.dataloaders.train.dataset.sl_cols

        feature_info = []
        for c in (self.sn_cols + self.sc_cols + self.sb_cols + self.sl_cols):
            info = [info for info in self.transformer._data_manipulation_info_list if info.column_name == c][0]
            feature_info.append(info)
        for c in (self.dataloaders.train.dataset.sc_cols + self.sl_cols + self.sn_cols):
            info = [info for info in self.transformer._data_manipulation_info_list if info.column_name == f'{c}_mask'][0]
            feature_info.append(info)
        self.feature_info = feature_info

        self.categorical_card = self.cfg.model.static_autoencoder.categorical_card
        self.diagnoses_prefix = self.cfg.preprocess.icd_code.diagnoses_prefix
        self.procedure_prefix = self.cfg.preprocess.icd_code.procedure_prefix

        self.use_gumbel = self.cfg.model.static_autoencoder.use_gumbel
        self.logit_threshold = self.cfg.model.static_autoencoder.logit_threshold
        self.conditional = self.cfg.model.static_autoencoder.conditional

    def get_beta(self, epoch, max_beta=0.1, annealing_epochs=50):
        if epoch >= annealing_epochs:
            return max_beta
        return max_beta * (epoch / annealing_epochs)

    def compute_loss(self,
                     x_num, x_cat, x_bin, x_listwise,
                     x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits,
                     mu, logvar, epoch, annealing_epochs=50,
                     x_mask=None, x_mask_logits=None, use_gumbel=True):
        device = x_num_hat.device

        # === Numerical Loss ===
        loss_num = F.mse_loss(x_num_hat, x_num, reduction='mean')

        # === Categorical Loss ===
        if x_cat:
            total_cat_loss = 0.0
            for target, logits in zip(x_cat, x_cat_logits):
                # target: (B,), logits: (B, card)
                total_cat_loss += F.cross_entropy(logits, target, reduction='mean')
            cat_loss = total_cat_loss / len(x_cat)
        else:
            cat_loss = torch.tensor(0.0, device=device)

        # === Binary Loss ===
        if x_bin is not None and x_bin_logits is not None and len(x_bin) > 0:
            total_bin_loss = 0.0
            for target, logits in zip(x_bin, x_bin_logits):
                if use_gumbel:
                    b_loss = self.focal_loss_from_probs(
                        logits, target.unsqueeze(-1).clamp(0.0, 1.0)
                    )
                    total_bin_loss += b_loss.mean()
                else:
                    total_bin_loss += F.binary_cross_entropy_with_logits(
                        logits, target.unsqueeze(-1).float(), reduction='mean'
                    )
            bin_loss = total_bin_loss / len(x_bin)
        else:
            bin_loss = torch.tensor(0.0, device=device)

        # === Listwise Loss ===
        if x_listwise is not None and x_listwise_logits is not None and len(x_listwise) > 0:
            total_listwise_loss = 0.0
            for target, logits in zip(x_listwise, x_listwise_logits):
                if use_gumbel:
                    l_loss = self.focal_loss_from_probs(
                        logits, target.clamp(0.0, 1.0)
                    )
                    l_loss = l_loss.mean()
                else:
                    l_loss = F.binary_cross_entropy_with_logits(
                        logits, target.float(), reduction='mean'
                    )
                total_listwise_loss += l_loss
            listwise_loss = total_listwise_loss / len(x_listwise)
        else:
            listwise_loss = torch.tensor(0.0, device=device)

        # === Mask loss ===
        if x_mask is not None and x_mask_logits is not None:
            total_mask_loss = 0.0
            for i in range(x_mask.shape[-1]):
                target = x_mask[:, i:i + 1]
                logits = x_mask_logits[i]

                if use_gumbel:
                    m_loss = self.focal_loss_from_probs(
                        logits, target.clamp(0.0, 1.0)
                    )
                    m_loss = m_loss.mean()
                else:
                    m_loss = F.binary_cross_entropy_with_logits(
                        logits, target.float(), reduction='mean'
                    )
                total_mask_loss += m_loss

            mask_loss = total_mask_loss / x_mask.shape[-1]
        else:
            mask_loss = torch.tensor(0.0, device=device)

        # === KL Loss ===
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if annealing_epochs is not None:
            kl_weight = self.get_beta(epoch, max_beta=0.1, annealing_epochs=annealing_epochs)
        else:
            kl_weight = 1.0
        weighted_kl = kl_weight * kl_loss

        total_loss = loss_num + cat_loss + bin_loss + listwise_loss + mask_loss + weighted_kl

        return (total_loss,
                loss_num,
                cat_loss,
                bin_loss,
                listwise_loss,
                mask_loss,
                kl_loss)

    def compute_semantic_loss(self, condition, condition_hat):
        if condition is None or condition_hat is None:
            return torch.tensor(0.0, device=self.device)

        target = condition.argmax(dim=-1)
        semantic_loss = F.cross_entropy(condition_hat, target, reduction='mean')

        return semantic_loss

    def focal_loss(self, logits, targets, pos_weight, alpha=0.25, gamma=2.0):
        # BCE with logits + focal modulation
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        modulating_factor = (1 - pt) ** gamma
        loss = alpha * modulating_factor * BCE_loss
        return loss

    def focal_loss_from_probs(self, probs, targets, alpha=0.25, gamma=2.0):
        BCE_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = alpha * (1 - pt) ** gamma * BCE_loss
        return loss

    @classmethod
    def onehot_to_index_with_mask(cls, sc_data, categorical_card):
        sb_data_converted = []
        sc_data_converted = []
        s_idx = 0
        for dim in categorical_card[:-2]:
            onehot = sc_data[:, s_idx:s_idx + dim]
            if dim > 1:
                indices = onehot.argmax(dim=-1)
                sc_data_converted.append(indices.to(torch.long))

            else:
                indices = onehot.squeeze(-1)
                sb_data_converted.append(indices.to(torch.long))
            s_idx += dim

        if len(sb_data_converted) > 0:
            return (sc_data_converted, torch.stack(sc_data_converted, dim=-1),
                    sb_data_converted, torch.stack(sb_data_converted, dim=-1))
        else:
            return (sc_data_converted, torch.stack(sc_data_converted, dim=-1),
                    None, None)

    def _unpack_batch(self, batch):
        sc_data = batch[0].to(self.device)
        sn_data = batch[2].to(self.device)
        sl_data = batch[4].to(self.device)
        static_mask = batch[6].to(self.device)
        condition = batch[8].to(self.device) if self.conditional else None
        return sc_data, sn_data, sl_data, static_mask, condition

    def _build_listwise_targets(self, sl_data: torch.Tensor):
        sl_target = []
        s_idx = 0
        for dim in self.categorical_card[-2:]:
            sl_target.append(sl_data[:, s_idx:s_idx + dim])
            s_idx += dim
        return sl_target

    def activate_x_hat(self, x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits):
        act_x_hat = []
        act_x_hat.append(x_num_hat)
        for sc_hat in x_cat_logits:
            _act_sc_hat = torch.softmax(sc_hat, dim=-1)
            _act_sc_hat = (_act_sc_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_sc_hat)

        for sb_hat in x_bin_logits:
            if self.use_gumbel:
                _act_sb_hat = sb_hat
            else:
                _act_sb_hat = torch.sigmoid(sb_hat)
                _act_sb_hat = (_act_sb_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_sb_hat)

        for sl_hat in x_listwise_logits:
            if self.use_gumbel:
                _act_sl_hat = sl_hat
            else:
                _act_sl_hat = torch.sigmoid(sl_hat)
                _act_sl_hat = (_act_sl_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_sl_hat)

        for sm_hat in x_mask_logits:
            if self.use_gumbel:
                _act_sm_hat = sm_hat
            else:
                _act_sm_hat = torch.sigmoid(sm_hat)
                _act_sm_hat = (_act_sm_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_sm_hat)
        act_x_hat = torch.cat(act_x_hat, dim=-1)

        return act_x_hat

    def get_target(self, sn_data, sc_data, sl_data, static_mask):
        sc_target = []
        sb_target = []
        s_idx = 0
        for i, dim in enumerate(self.categorical_card[:-2]):
            if dim > 1:
                sc_target.append(sc_data[:, s_idx:s_idx + dim])
            else:
                sb_target.append(sc_data[:, s_idx:s_idx + dim])
            s_idx += dim
        sc_target = torch.cat(sc_target, dim=-1)
        if len(sb_target) > 0:
            sb_target = torch.cat(sb_target, dim=-1)
        else:
            sb_target = torch.empty((sc_target.shape[0], 0), device=sc_target.device)

        data = torch.cat([sn_data, sc_target, sb_target, sl_data, static_mask], dim=-1)

        return data

    def run_epochs(self):
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.model.train()
            self.train_one_epoch(epoch)
            self.model.eval()
            self.validate_one_epoch(epoch)

            if epoch % self.cfg.train.general.eval_freq == 0:
                self.eval_one_epoch(epoch)

            if self.scheduler is not None:
                stop_flag, self.scheduler_flag = self.train_watcher.check(epoch,
                                                                          self.validation_loss[self.target_key][-1])
                if stop_flag:
                    break

                if self.scheduler_flag:
                    self.scheduler.step()

            save_ckpt(
                cfg=self.cfg,
                epoch=epoch,
                validation_loss=self.validation_loss['Total_Loss'],
                states={'epoch': epoch,
                        'arch': self.cfg.log.time,
                        'model': self.model,
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None
                        },
                rank=self.rank,
                save_condition='lower'
            )
            self.wandb.log(self.train_loss, self.validation_loss, epoch)

        self.wandb.cleanup()

    def train_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        train_iterator = self._set_iterator(self.dataloaders.train, epoch, mode='Train')

        for batch in train_iterator:
            self.optimizer.zero_grad()

            sc_data, sn_data, sl_data, static_mask, condition = self._unpack_batch(batch)
            (sc_data_target, sc_data_converted,
             sb_data_target, sb_data_converted) = self.onehot_to_index_with_mask(sc_data, self.categorical_card)

            (x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits,
             mu, logvar, z, condition_hat) = self.model(
                x_num=sn_data,
                x_cat=sc_data_converted,
                x_listwise=sl_data,
                x_bin=sb_data_converted,
                x_mask=static_mask,
                use_gumbel=self.use_gumbel
            )

            sl_target = self._build_listwise_targets(sl_data)

            loss, loss_num, cat_loss, bin_loss, listwise_loss, mask_loss, kl_loss = self.compute_loss(
                x_num=sn_data,
                x_cat=sc_data_target,
                x_bin=sb_data_target,
                x_listwise=sl_target,
                x_num_hat=x_num_hat,
                x_cat_logits=x_cat_logits,
                x_bin_logits=x_bin_logits,
                x_listwise_logits=x_listwise_logits,
                x_mask=static_mask,
                x_mask_logits=x_mask_logits,
                use_gumbel=self.use_gumbel,
                mu=mu, logvar=logvar,
                epoch=epoch
            )
            semantic_loss = self.compute_semantic_loss(condition, condition_hat)

            loss += semantic_loss

            total_losses['Total_Loss'] += loss.item()
            total_losses['MSE_Loss'] += loss_num.item()
            total_losses['CE_Loss'] += cat_loss.item()
            total_losses['BCE_Loss'] += bin_loss.item()
            total_losses['Listwise_Loss'] += listwise_loss.item()
            total_losses['Mask_Loss'] += mask_loss.item()
            total_losses['KL_Loss'] += kl_loss.item()
            total_losses['Semantic_Loss'] += semantic_loss.item()

            loss.backward()
            self.optimizer.step()
            self._set_iterator_postfix(train_iterator, total_losses['Total_Loss'])

        for key in self.loss_keys:
            self.train_loss[key].append(total_losses[key] / len(self.dataloaders.train))

    def validate_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        validation_iterator = self._set_iterator(self.dataloaders.valid, epoch, mode='val')

        with torch.no_grad():
            for batch in validation_iterator:
                sc_data, sn_data, sl_data, static_mask, condition = self._unpack_batch(batch)

                (sc_data_target, sc_data_converted,
                 sb_data_target, sb_data_converted) = self.onehot_to_index_with_mask(sc_data, self.categorical_card)

                (x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits,
                 mu, logvar, z, condition_hat) = self.model(
                    x_num=sn_data,
                    x_cat=sc_data_converted,
                    x_listwise=sl_data,
                    x_bin=sb_data_converted,
                    x_mask=static_mask,
                    use_gumbel=self.use_gumbel
                )

                sl_target = self._build_listwise_targets(sl_data)

                loss, loss_num, cat_loss, bin_loss, listwise_loss, mask_loss, kl_loss = self.compute_loss(
                    x_num=sn_data,
                    x_cat=sc_data_target,
                    x_bin=sb_data_target,
                    x_listwise=sl_target,
                    x_num_hat=x_num_hat,
                    x_cat_logits=x_cat_logits,
                    x_bin_logits=x_bin_logits,
                    x_listwise_logits=x_listwise_logits,
                    x_mask=static_mask,
                    x_mask_logits=x_mask_logits,
                    use_gumbel=self.use_gumbel,
                    mu=mu, logvar=logvar,
                    epoch=epoch
                )
                semantic_loss = self.compute_semantic_loss(condition, condition_hat)

                loss += semantic_loss

                total_losses['Total_Loss'] += loss.item()
                total_losses['MSE_Loss'] += loss_num.item()
                total_losses['CE_Loss'] += cat_loss.item()
                total_losses['BCE_Loss'] += bin_loss.item()
                total_losses['Listwise_Loss'] += listwise_loss.item()
                total_losses['Mask_Loss'] += mask_loss.item()
                total_losses['KL_Loss'] += kl_loss.item()
                total_losses['Semantic_Loss'] += semantic_loss.item()

                self._set_iterator_postfix(validation_iterator, total_losses['Total_Loss'])

            for key in self.loss_keys:
                self.validation_loss[key].append(total_losses[key] / len(self.dataloaders.valid))

    def eval_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        test_iterator = self._set_iterator(self.dataloaders.test, epoch, mode='test')

        if not os.path.exists(self.cfg.path.plot_file_path):
            os.makedirs(self.cfg.path.plot_file_path, exist_ok=True)

        with torch.no_grad():
            target = []
            data_hat = []
            conditions = []
            latent_vectors = []
            for batch in test_iterator:
                sc_data, sn_data, sl_data, static_mask, condition = self._unpack_batch(batch)

                (sc_data_target, sc_data_converted,
                 sb_data_target, sb_data_converted) = self.onehot_to_index_with_mask(sc_data, self.categorical_card)

                (x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits,
                 mu, logvar, z, condition_hat) = self.model(
                    x_num=sn_data,
                    x_cat=sc_data_converted,
                    x_listwise=sl_data,
                    x_bin=sb_data_converted,
                    x_mask=static_mask,
                    use_gumbel=self.use_gumbel,
                    hard=True
                )

                sl_target = self._build_listwise_targets(sl_data)

                loss, loss_num, cat_loss, bin_loss, listwise_loss, mask_loss, kl_loss = self.compute_loss(
                    x_num=sn_data,
                    x_cat=sc_data_target,
                    x_bin=sb_data_target,
                    x_listwise=sl_target,
                    x_num_hat=x_num_hat,
                    x_cat_logits=x_cat_logits,
                    x_bin_logits=x_bin_logits,
                    x_listwise_logits=x_listwise_logits,
                    x_mask=static_mask,
                    x_mask_logits=x_mask_logits,
                    use_gumbel=self.use_gumbel,
                    mu=mu, logvar=logvar,
                    epoch=epoch
                )
                semantic_loss = self.compute_semantic_loss(condition, condition_hat)

                loss += semantic_loss

                total_losses['Total_Loss'] += loss.item()
                total_losses['MSE_Loss'] += loss_num.item()
                total_losses['CE_Loss'] += cat_loss.item()
                total_losses['BCE_Loss'] += bin_loss.item()
                total_losses['Listwise_Loss'] += listwise_loss.item()
                total_losses['Mask_Loss'] += mask_loss.item()
                total_losses['KL_Loss'] += kl_loss.item()
                total_losses['Semantic_Loss'] += semantic_loss.item()
                self._set_iterator_postfix(test_iterator, total_losses['Total_Loss'])

                act_x_hat = self.activate_x_hat(x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits)
                data = self.get_target(sn_data, sc_data, sl_data, static_mask)
                target.append(data)
                data_hat.append(act_x_hat)
                conditions.append(condition)
                latent_vectors.append(z)

            target = torch.concatenate(target, dim=0)
            data_hat = torch.concatenate(data_hat, dim=0)
            conditions = torch.concatenate(conditions, dim=0) if self.conditional else None
            latent_vectors = torch.concatenate(latent_vectors, dim=0)

            feature_dim = target.shape[-1]
            target = self.transformer.inverse_transform(target.detach().cpu().numpy(), self.feature_info)
            data_hat = self.transformer.inverse_transform(data_hat.detach().cpu().numpy(), self.feature_info)

            latent_vectors = latent_vectors.detach().cpu().numpy()
            conditions_np = conditions.detach().cpu().numpy()
            conditions_indices = np.argmax(conditions_np, axis=-1)
            conditions = pd.DataFrame(conditions_np, columns=['survived', 'icu_mortality', 'hospital_mortality'])

            eval_numerical_features(cols=self.sn_cols, real=target, synthetic=data_hat,
                                    labels=['Real', 'Reconstructed'],
                                    epoch=epoch, save_path=self.cfg.path.plot_file_path)

            icd_d_cols = [col for col in target.columns if
                          (col.startswith(self.diagnoses_prefix)) and ('mask' not in col)]
            icd_p_cols = [col for col in target.columns if
                          (col.startswith(self.procedure_prefix)) and ('mask' not in col)]
            eval_categorical_features(cols=self.sc_cols + self.sb_cols + icd_d_cols + icd_p_cols, real=target,
                                      synthetic=data_hat,
                                      labels=['Real', 'Reconstructed'],
                                      epoch=epoch, save_path=self.cfg.path.plot_file_path)

            scatterplot_dimension_wise_probability(target[icd_d_cols + icd_p_cols].dropna(),
                                                   data_hat[icd_d_cols + icd_p_cols].dropna(),
                                                   'Real',
                                                   'Reconstructed',
                                                   save_path=os.path.join(self.cfg.path.plot_file_path,
                                                                          f'dimension_wise_probability_icd_{epoch}.png'))

            mask_cols = [col for col in target.columns if 'mask' in col]
            scatterplot_dimension_wise_probability(target[mask_cols].dropna(),
                                                   data_hat[mask_cols].dropna(),
                                                   'Real',
                                                   'Reconstructed',
                                                   save_path=os.path.join(self.cfg.path.plot_file_path,
                                                                          f'dimension_wise_probability_mask_{epoch}.png'))

            pearson_pairwise_correlation_comparison(target[self.sn_cols],
                                                    data_hat[self.sn_cols],
                                                    figsize=(30, 10),
                                                    plot_file_path=os.path.join(self.cfg.path.plot_file_path,
                                                                                f'pearson_correlation_comparison_{epoch}.png'))
            pearson_pairwise_correlation_comparison(target[self.sc_cols + self.sb_cols],
                                                    data_hat[self.sc_cols + self.sb_cols],
                                                    figsize=(30, 10),
                                                    plot_file_path=os.path.join(self.cfg.path.plot_file_path,
                                                                                f'cramers_v_correlation_comparison_{epoch}.png'),
                                                    categorical=True)
            pearson_pairwise_correlation_comparison(target[icd_d_cols],
                                                    data_hat[icd_d_cols],
                                                    figsize=(30, 10),
                                                    plot_file_path=os.path.join(self.cfg.path.plot_file_path,
                                                                                f'cramers_v_correlation_comparison_icd_d_{epoch}.png'),
                                                    categorical=True)
            pearson_pairwise_correlation_comparison(target[icd_p_cols],
                                                    data_hat[icd_p_cols],
                                                    figsize=(30, 10),
                                                    plot_file_path=os.path.join(self.cfg.path.plot_file_path,
                                                                                f'cramers_v_correlation_comparison_icd_p_{epoch}.png'),
                                                    categorical=True)

            # evaluate condition
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_results = tsne.fit_transform(latent_vectors)  # [n_samples, 2]
            condition_labels = {0: 'Survived',
                                1: 'ICU mortality',
                                2: 'Hospital mortality'}
            colors = {
                0: 'grey',
                1: 'red',
                2: 'green'
            }

            plt.figure(figsize=(8, 6))
            for cond_value in np.unique(conditions_indices):
                idx = conditions_indices == cond_value
                plt.scatter(
                    tsne_results[idx, 0],
                    tsne_results[idx, 1],
                    s=15,
                    alpha=0.6,
                    label=condition_labels.get(cond_value, f'Class {cond_value}'),
                    color=colors.get(cond_value, None)
                )

            plt.xlabel('tSNE 1')
            plt.ylabel('tSNE 2')
            plt.legend()
            plt.title('t-SNE of latent vectors by condition')
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg.path.plot_file_path, f'tsne_by_condition_{epoch}.png'))
            plt.close()

            target = pd.concat([target, conditions], axis=1)
            data_hat = pd.concat([data_hat, conditions], axis=1)

            target_cols = 'icu_mortality'
            for col in (self.sc_cols):
                if col == target_cols:
                    continue

                target['type'] = 'Real'
                data_hat['type'] = 'Synthetic'
                data = pd.concat([target, data_hat], axis=0)
                data = data.reset_index(drop=True)
                plt.figure(figsize=(8, 6))
                g = sns.catplot(
                    data=data,
                    x=col,
                    y=target_cols,
                    col='type',
                    kind='bar',
                    height=6,
                    aspect=0.5
                )

                for ax in g.axes.flat:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.cfg.path.plot_file_path, f'{col}_distribution_by_{target_cols}_{epoch}.png'))

            for col in (self.sn_cols):
                target['type'] = 'Real'
                data_hat['type'] = 'Synthetic'
                data = pd.concat([target, data_hat], axis=0)
                data = data.reset_index(drop=True)
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=data, x='type', y=col, hue=target_cols)
                plt.legend(loc='upper right')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.cfg.path.plot_file_path, f'{col}_distribution_by_{target_cols}_{epoch}.png'))

            for key in self.loss_keys:
                self.test_loss[key].append(total_losses[key] / len(self.dataloaders.test))


def trainer_main(cols: List[str] = None):
    config_manager.load_config()
    cfg = config_manager.config
    cfg.log.wandb.name = f'{cfg.log.time}_{cfg.log.ipaddr}'
    cfg.log.wandb.tags = ['StaticAE']

    lock_seed(seed=cfg.seed, multi_gpu=False, activate_cudnn=False)

    dataset_name = cfg.dataset.dataset_name
    fname = cfg.preprocess.preprocess_fname_suffix
    train_ratio, test_ratio = cfg.preprocess.train_valid_test_split_ratio
    dataset_fname = f'{dataset_name}_{fname}_{int(train_ratio * 10)}.h5'

    train_dataset = CustomDataset(cfg=cfg,
                                  dataset_name=dataset_name,
                                  dataset_fname=dataset_fname,
                                  mode='train',
                                  condition_col=cfg.data.condition_col,
                                  static_cols=cols)
    validation_dataset = CustomDataset(cfg=cfg,
                                       dataset_name=dataset_name,
                                       dataset_fname=dataset_fname,
                                       mode='val',
                                       condition_col=cfg.data.condition_col,
                                       static_cols=cols)
    test_dataset = CustomDataset(cfg=cfg,
                                 dataset_name=dataset_name,
                                 dataset_fname=dataset_fname,
                                 mode='test',
                                 condition_col=cfg.data.condition_col,
                                 static_cols=cols)

    train_sampler = None
    validation_sampler = None

    sn_cols = train_dataset.sn_cols
    sc_cols = train_dataset.sc_cols
    sl_cols = train_dataset.sl_cols
    categorical_feature_info = []
    for c in (sc_cols + sl_cols):
        info = \
        [info for info in train_dataset.static_transformer._data_manipulation_info_list if info.column_name == c][0]
        categorical_feature_info.append(info)
    categorical_feature_out_dims = [info.output_dimensions for info in categorical_feature_info]
    set_cfg(cfg, 'model.static_autoencoder.categorical_card', categorical_feature_out_dims)
    set_cfg(cfg, 'model.static_autoencoder.num_numerical', len(train_dataset.sn_cols))

    model = build_model(cfg.model.static_autoencoder, device=torch.device(f'cuda:{cfg.device_num}'))
    if cfg.train.general.init_weight:
        initialize_weights(model)

    if cfg.train.general.distributed:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    loss_fn = set_loss_fn(cfg.train.loss)
    optimizer = set_optimizer(model.parameters(), cfg.train.optimizer, apply_lookahead=False)
    scheduler = set_scheduler(optimizer, cfg.train.scheduler)

    trainer = StaticVAETrainer(config=cfg,
                               rank=cfg.device_num,
                               world_size=world_size,
                               model=model,
                               collate_fn=None,
                               loss_fn=loss_fn,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               train_dataset=train_dataset,
                               validation_dataset=validation_dataset,
                               test_dataset=test_dataset,
                               train_sampler=train_sampler,
                               validation_sampler=validation_sampler)
    trainer.run_epochs()


if __name__ == '__main__':
    cols = None
    trainer_main(cols=cols)

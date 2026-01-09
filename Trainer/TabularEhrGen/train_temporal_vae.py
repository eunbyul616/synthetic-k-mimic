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

from Utils.train import save_ckpt, initialize_weights
from Utils.reproducibility import lock_seed
from Utils.train import set_loss_fn, set_scheduler, set_optimizer
from Utils.namespace import set_cfg

from Models.TabularEhrGen.TemporalVAE import build_model

from Visualization import *
from Visualization.timeseries import *
from Evaluation.DistributionSimilarity.correlation_comparison import *

import config_manager

import torch
import torch.nn.functional as F


class TemporalVAETrainer(Trainer):
    def __init__(self, config, rank, **kwargs):
        super(TemporalVAETrainer, self).__init__(config=config, rank=rank, **kwargs)

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

        self.temporal_transformer = self.dataloaders.train.dataset.temporal_transformer

        tc_cols = []
        tb_cols = []
        for c in self.dataloaders.train.dataset.tc_cols:
            info = [info for info in self.temporal_transformer._data_manipulation_info_list if info.column_name == c][0]
            dim = info.output_dimensions
            if dim == 1:
                tb_cols.append(c)
            else:
                tc_cols.append(c)

        self.tb_cols = tb_cols
        self.tc_cols = tc_cols
        self.tn_cols = self.dataloaders.train.dataset.tn_cols
        self.tl_cols = self.dataloaders.train.dataset.tl_cols

        temporal_feature_info = []
        for c in (self.tn_cols + self.tc_cols + self.tb_cols + self.tl_cols):
            info = [info for info in self.temporal_transformer._data_manipulation_info_list if info.column_name == c][0]
            temporal_feature_info.append(info)
        for c in (self.dataloaders.train.dataset.tc_cols + self.tl_cols + self.tn_cols):
            info = [info for info in self.temporal_transformer._data_manipulation_info_list if
                    info.column_name == f'{c}_mask'][0]
            temporal_feature_info.append(info)
        self.temporal_feature_info = temporal_feature_info

        self.temporal_categorical_card = self.cfg.model.temporal_autoencoder.categorical_card

        self.diagnoses_prefix = self.cfg.preprocess.icd_code.diagnoses_prefix
        self.procedure_prefix = self.cfg.preprocess.icd_code.procedure_prefix
        self.proc_prefix = self.cfg.preprocess.proc_prefix

        self.use_gumbel = self.cfg.model.static_autoencoder.use_gumbel
        self.logit_threshold = self.cfg.model.static_autoencoder.logit_threshold
        self.conditional = self.cfg.model.static_autoencoder.conditional

        self.lambda_sem = 0.5

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
        loss_num = F.mse_loss(x_num_hat, x_num, reduction='none')
        mask = (x_num != -1).all(dim=-1).float()  # (B, T)
        loss_num = (loss_num.sum(dim=-1) * mask).sum() / (mask.sum() + 1e-8)

        # === Categorical Loss ===
        if x_cat:
            total_cat_loss = 0.0
            padding_indices = [card for card in self.temporal_categorical_card[:-1] if card > 1]
            for i, (target, logits) in enumerate(zip(x_cat, x_cat_logits)):
                # target: (B, T), logits: (B, T, card)
                padding_idx = padding_indices[i]
                total_cat_loss += F.cross_entropy(
                    logits.transpose(1, 2),  # (B, card, T)
                    target,
                    ignore_index=padding_idx,
                )
            cat_loss = total_cat_loss / len(x_cat)
        else:
            cat_loss = torch.tensor(0.0, device=device)

        # === Binary Loss ===
        if x_bin is not None and x_bin_logits is not None and len(x_bin) > 0:
            total_bin_loss = 0.0
            padding_idx = -1
            for target, logits in zip(x_bin, x_bin_logits):
                # target: (B, T), logits: (B, T, 1)
                b_loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), target.float(), reduction='none'
                )
                mask = (target != padding_idx).float()
                b_loss = (b_loss * mask).sum() / (mask.sum() + 1e-8)
                total_bin_loss += b_loss
            bin_loss = total_bin_loss / len(x_bin)
        else:
            bin_loss = torch.tensor(0.0, device=device)

        # === Listwise Loss ===
        if x_listwise is not None and x_listwise_logits is not None and len(x_listwise) > 0:
            total_listwise_loss = 0.0
            padding_idx = -1
            for target, logits in zip(x_listwise, x_listwise_logits):
                # target, logits: (B, T, D_l)
                if use_gumbel:
                    l_loss = self.focal_loss_from_probs(
                        logits, target.clamp(0.0, 1.0)
                    )
                else:
                    l_loss = F.binary_cross_entropy_with_logits(
                        logits, target.float(), reduction='none'
                    )
                mask = (target != padding_idx).float()
                l_loss = (l_loss * mask).sum() / (mask.sum() + 1e-8)
                total_listwise_loss += l_loss
            listwise_loss = total_listwise_loss / len(x_listwise)
        else:
            listwise_loss = torch.tensor(0.0, device=device)

        # === Mask Loss ===
        if x_mask is not None and x_mask_logits is not None:
            total_mask_loss = 0.0
            padding_idx = -1
            for i, logits in enumerate(x_mask_logits):
                # target, logits: (B, T, 1)
                target = x_mask[:, :, i:i + 1]

                if use_gumbel:
                    m_loss = self.focal_loss_from_probs(
                        logits, target.clamp(0.0, 1.0)
                    )
                else:
                    m_loss = F.binary_cross_entropy_with_logits(
                        logits, target.float(), reduction='none'
                    )

                mask = (target != padding_idx).float()
                m_loss = (m_loss * mask).sum() / (mask.sum() + 1e-8)
                total_mask_loss += m_loss

            mask_loss = total_mask_loss / len(x_mask_logits)
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

        target = torch.argmax(condition, dim=-1)
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
    def static_onehot_to_index_with_mask(cls, sc_data, categorical_card):
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

    @classmethod
    def temporal_onehot_to_index_with_mask(cls, tc_data, categorical_card):
        tb_data_converted = []
        tc_data_converted = []
        s_idx = 0
        for dim in categorical_card[:-1]:
            onehot = tc_data[:, :, s_idx:s_idx + dim]
            values, indices = torch.max(onehot, dim=-1)
            is_masked = values == -1
            if dim > 1:
                indices = onehot.argmax(dim=-1)
                indices[is_masked] = dim
                tc_data_converted.append(indices.to(torch.long))
            else:
                indices = onehot.squeeze(-1)
                indices[is_masked] = -1
                tb_data_converted.append(indices.to(torch.long))
            s_idx += dim

        if len(tb_data_converted) > 0:
            return (tc_data_converted, torch.stack(tc_data_converted, dim=-1),
                    tb_data_converted, torch.stack(tb_data_converted, dim=-1))
        else:
            return (tc_data_converted, torch.stack(tc_data_converted, dim=-1),
                    None, None)

    def _unpack_static_batch(self, batch):
        sc_data = batch[0].to(self.device)
        sn_data = batch[2].to(self.device)
        sl_data = batch[4].to(self.device)
        static_mask = batch[6].to(self.device)
        return sc_data, sn_data, sl_data, static_mask

    def _unpack_temporal_batch(self, batch):
        tc_data = batch[1].to(self.device)
        tn_data = batch[3].to(self.device)
        tl_data = batch[5].to(self.device)
        temporal_mask = batch[7].to(self.device)
        condition = batch[8].to(self.device) if self.conditional else None
        return tc_data, tn_data, tl_data, temporal_mask, condition

    def _build_temporal_listwise_targets(self, tl_data: torch.Tensor):
        tl_target = []
        s_idx = 0
        for dim in self.temporal_categorical_card[-1:]:
            tl_target.append(tl_data[:, :, s_idx:s_idx + dim])
            s_idx += dim
        return tl_target

    def activate_static_hat(self, x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits):
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

    def activate_x_hat(self, x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logits, x_mask_logits):
        act_x_hat = []
        act_x_hat.append(x_num_hat)
        for tc_hat in x_cat_logits:
            _act_tc_hat = torch.softmax(tc_hat, dim=-1)
            _act_tc_hat = (_act_tc_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_tc_hat)

        for tb_hat in x_bin_logits:
            _act_tb_hat = torch.sigmoid(tb_hat)
            _act_tb_hat = (_act_tb_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_tb_hat)

        for tl_hat in x_listwise_logits:
            if self.use_gumbel:
                _act_tl_hat = tl_hat
            else:
                _act_tl_hat = torch.sigmoid(tl_hat)
                _act_tl_hat = (_act_tl_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_tl_hat)

        for tm_hat in x_mask_logits:
            if self.use_gumbel:
                _act_tm_hat = tm_hat
            else:
                _act_tm_hat = torch.sigmoid(tm_hat)
                _act_tm_hat = (_act_tm_hat >= self.logit_threshold).float()
            act_x_hat.append(_act_tm_hat)
        act_x_hat = torch.cat(act_x_hat, dim=-1)

        return act_x_hat

    def get_target(self, tn_data, tc_data, tl_data, temporal_mask):
        tc_target = []
        tb_target = []
        s_idx = 0
        for i, dim in enumerate(self.temporal_categorical_card[:-1]):
            if dim > 1:
                tc_target.append(tc_data[:, :, s_idx:s_idx + dim])
            else:
                tb_target.append(tc_data[:, :, s_idx:s_idx + dim])
            s_idx += dim

        tc_target = torch.cat(tc_target, dim=-1)
        if len(tb_target) > 0:
            tb_target = torch.cat(tb_target, dim=-1)
        else:
            tb_target = torch.empty(tc_target.shape[0], tc_target.shape[1], 0, device=tc_target.device)

        data = torch.cat([tn_data, tc_target, tb_target, tl_data, temporal_mask], dim=-1)

        return data

    def run_epochs(self):
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.model.train()
            self.train_one_epoch(epoch)
            self.model.eval()
            self.validate_one_epoch(epoch)

            if epoch > 0 and epoch % self.cfg.train.general.eval_freq == 0:
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

    def train_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}

        train_iterator = self._set_iterator(self.dataloaders.train, epoch, mode='Train')

        for batch in train_iterator:
            self.optimizer.zero_grad()

            tc_data, tn_data, tl_data, temporal_mask, condition = self._unpack_temporal_batch(batch)
            (tc_data_target, tc_data_converted,
             tb_data_target, tb_data_converted) = self.temporal_onehot_to_index_with_mask(tc_data,
                                                                                          self.temporal_categorical_card)

            (x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logit, x_mask_logits,
             mu, logvar, z, condition_hat) = self.model(
                x_num=tn_data,
                x_cat=tc_data_converted,
                x_listwise=tl_data,
                x_bin=tb_data_converted,
                x_mask=temporal_mask,
                use_gumbel=self.use_gumbel
            )

            tl_target = self._build_temporal_listwise_targets(tl_data)

            (loss, loss_num, cat_loss, bin_loss, listwise_loss, mask_loss, kl_loss) = self.compute_loss(
                x_num=tn_data,
                x_cat=tc_data_target,
                x_bin=tb_data_target,
                x_listwise=tl_target,
                x_num_hat=x_num_hat,
                x_cat_logits=x_cat_logits,
                x_bin_logits=x_bin_logits,
                x_listwise_logits=x_listwise_logit,
                x_mask=temporal_mask,
                x_mask_logits=x_mask_logits,
                mu=mu,
                logvar=logvar,
                epoch=epoch,
                use_gumbel=self.use_gumbel
            )
            semantic_loss = self.compute_semantic_loss(condition, condition_hat)
            loss = loss + self.lambda_sem * semantic_loss

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
                tc_data, tn_data, tl_data, temporal_mask, condition = self._unpack_temporal_batch(batch)

                (tc_data_target, tc_data_converted,
                 tb_data_target, tb_data_converted) = self.temporal_onehot_to_index_with_mask(tc_data,
                                                                                              self.temporal_categorical_card)

                (x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logit, x_mask_logits,
                 mu, logvar, z, condition_hat) = self.model(
                    x_num=tn_data,
                    x_cat=tc_data_converted,
                    x_listwise=tl_data,
                    x_bin=tb_data_converted,
                    x_mask=temporal_mask,
                    use_gumbel=self.use_gumbel,
                )

                tl_target = self._build_temporal_listwise_targets(tl_data)

                (loss, loss_num, cat_loss, bin_loss, listwise_loss, mask_loss, kl_loss) = self.compute_loss(
                    x_num=tn_data,
                    x_cat=tc_data_target,
                    x_bin=tb_data_target,
                    x_listwise=tl_target,
                    x_num_hat=x_num_hat,
                    x_cat_logits=x_cat_logits,
                    x_bin_logits=x_bin_logits,
                    x_listwise_logits=x_listwise_logit,
                    x_mask=temporal_mask,
                    x_mask_logits=x_mask_logits,
                    mu=mu,
                    logvar=logvar,
                    epoch=epoch,
                    use_gumbel=self.use_gumbel
                )
                semantic_loss = self.compute_semantic_loss(condition, condition_hat)

                loss = loss + self.lambda_sem * semantic_loss

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
                tc_data, tn_data, tl_data, temporal_mask, condition = self._unpack_temporal_batch(batch)

                seq_len = tc_data.shape[1]

                (tc_data_target, tc_data_converted,
                 tb_data_target, tb_data_converted) = self.temporal_onehot_to_index_with_mask(tc_data,
                                                                                              self.temporal_categorical_card)

                (x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logit, x_mask_logits,
                 mu, logvar, z, condition_hat) = self.model(
                    x_num=tn_data,
                    x_cat=tc_data_converted,
                    x_listwise=tl_data,
                    x_bin=tb_data_converted,
                    x_mask=temporal_mask,
                    use_gumbel=self.use_gumbel,
                    hard=True,
                )

                tl_target = self._build_temporal_listwise_targets(tl_data)

                (loss, loss_num, cat_loss, bin_loss, listwise_loss, mask_loss, kl_loss) = self.compute_loss(
                    x_num=tn_data,
                    x_cat=tc_data_target,
                    x_bin=tb_data_target,
                    x_listwise=tl_target,
                    x_num_hat=x_num_hat,
                    x_cat_logits=x_cat_logits,
                    x_bin_logits=x_bin_logits,
                    x_listwise_logits=x_listwise_logit,
                    x_mask=temporal_mask,
                    x_mask_logits=x_mask_logits,
                    mu=mu,
                    logvar=logvar,
                    epoch=epoch,
                    use_gumbel=self.use_gumbel
                )
                semantic_loss = self.compute_semantic_loss(condition, condition_hat)

                loss = loss + self.lambda_sem * semantic_loss

                total_losses['Total_Loss'] += loss.item()
                total_losses['MSE_Loss'] += loss_num.item()
                total_losses['CE_Loss'] += cat_loss.item()
                total_losses['BCE_Loss'] += bin_loss.item()
                total_losses['Listwise_Loss'] += listwise_loss.item()
                total_losses['Mask_Loss'] += mask_loss.item()
                total_losses['KL_Loss'] += kl_loss.item()
                total_losses['Semantic_Loss'] += semantic_loss.item()

                self._set_iterator_postfix(test_iterator, total_losses['Total_Loss'])

                act_x_hat = self.activate_x_hat(x_num_hat, x_cat_logits, x_bin_logits, x_listwise_logit, x_mask_logits)
                data = self.get_target(tn_data, tc_data, tl_data, temporal_mask)
                target.append(data)
                data_hat.append(act_x_hat)
                conditions.append(condition)
                latent_vectors.append(z)

            target = torch.concatenate(target, dim=0)
            data_hat = torch.concatenate(data_hat, dim=0)
            conditions = torch.concatenate(conditions, dim=0) if self.conditional else None
            latent_vectors = torch.concatenate(latent_vectors, dim=0)

            mask = (target != -1)
            feature_dim = target.shape[-1]
            target = self.temporal_transformer.inverse_transform(target.view(-1, feature_dim).detach().cpu().numpy(),
                                                                 self.temporal_feature_info)
            data_hat = self.temporal_transformer.inverse_transform(
                data_hat.view(-1, feature_dim).detach().cpu().numpy(), self.temporal_feature_info)

            mask = mask.view(-1, feature_dim).detach().cpu().numpy()
            target = target[mask.all(axis=1)]
            data_hat = data_hat[mask.all(axis=1)]

            latent_vectors = latent_vectors.detach().cpu().numpy()

            conditions = conditions.detach().cpu().numpy()
            conditions_indices = conditions.argmax(axis=-1)
            conditions = pd.DataFrame(np.repeat(conditions, seq_len, axis=0),
                                      columns=['survived', 'icu_mortality', 'hospital_mortality'])

            proc_cols = [col for col in target.columns if (col.startswith(self.proc_prefix)) and ('mask' not in col)]
            # eval_numerical_features(cols=self.tn_cols, real=target, synthetic=data_hat,
            #                         labels=['Real', 'Reconstructed'],
            #                         epoch=epoch, save_path=self.cfg.path.plot_file_path)
            eval_categorical_features(cols=self.tc_cols + self.tb_cols + proc_cols, real=target, synthetic=data_hat,
                                      labels=['Real', 'Reconstructed'],
                                      epoch=epoch, save_path=self.cfg.path.plot_file_path)

            scatterplot_dimension_wise_probability(target[proc_cols].dropna(),
                                                   data_hat[proc_cols].dropna(),
                                                   'Real',
                                                   'Reconstructed',
                                                   save_path=os.path.join(self.cfg.path.plot_file_path,
                                                                          f'dimension_wise_probability_procedureevents_{epoch}.png'))

            if len(self.tb_cols) > 0:
                scatterplot_dimension_wise_probability(target[self.tb_cols].dropna(),
                                                       data_hat[self.tb_cols].dropna(),
                                                       'Real',
                                                       'Reconstructed',
                                                       save_path=os.path.join(self.cfg.path.plot_file_path,
                                                                              f'dimension_wise_probability_{epoch}.png'))

            mask_cols = [col for col in target.columns if 'mask' in col]
            scatterplot_dimension_wise_probability(target[mask_cols].dropna(),
                                                   data_hat[mask_cols].dropna(),
                                                   'Real',
                                                   'Reconstructed',
                                                   save_path=os.path.join(self.cfg.path.plot_file_path,
                                                                          f'dimension_wise_probability_mask_{epoch}.png'))

            pearson_pairwise_correlation_comparison(target[self.tn_cols],
                                                    data_hat[self.tn_cols],
                                                    figsize=(30, 10),
                                                    plot_file_path=os.path.join(self.cfg.path.plot_file_path,
                                                                                f'pearson_correlation_comparison_{epoch}.png'))
            pearson_pairwise_correlation_comparison(target[self.tc_cols + self.tb_cols],
                                                    data_hat[self.tc_cols + self.tb_cols],
                                                    figsize=(30, 10),
                                                    plot_file_path=os.path.join(self.cfg.path.plot_file_path,
                                                                                f'cramers_v_correlation_comparison_{epoch}.png'),
                                                    categorical=True)
            pearson_pairwise_correlation_comparison(target[proc_cols],
                                                    data_hat[proc_cols],
                                                    figsize=(30, 10),
                                                    plot_file_path=os.path.join(self.cfg.path.plot_file_path,
                                                                                f'cramers_v_correlation_comparison_proc_{epoch}.png'),
                                                    categorical=True)

            # for col in self.tn_cols:
            #     _col = col.replace('/', ' ').replace('>', ' ')
            #     vis_acf(data1=target[[col]].dropna(), label1='Real', data2=data_hat[[col]].dropna(),
            #             label2='Synthetic',
            #             title=f'{col}',
            #             save_path=os.path.join(self.cfg.path.plot_file_path, f'acf_{_col}_{epoch}.png'))

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

            target = self.dataloaders.test.dataset.temporal_data.reset_index(drop=True)
            target = pd.concat([target, conditions], axis=1)
            data_hat = pd.concat([data_hat, conditions], axis=1)

            target_cols = 'icu_mortality'
            for col in (self.tc_cols):
                if col == target_cols:
                    continue

                target['type'] = 'Real'
                data_hat['type'] = 'Synthetic'
                data = pd.concat([target, data_hat], axis=0)
                data = data.reset_index(drop=True)
                plt.figure(figsize=(8, 6))
                g = sns.catplot(data=data, x=col, y=target_cols, col='type',
                                kind='bar', height=6, aspect=0.5)
                for ax in g.axes.flat:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.cfg.path.plot_file_path, f'{col}_distribution_by_{target_cols}_{epoch}.png'))

            for col in (self.tn_cols):
                target['type'] = 'Real'
                data_hat['type'] = 'Synthetic'
                data = pd.concat([target, data_hat], axis=0)
                data = data.reset_index(drop=True)
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=data, x='type', y=col, hue=target_cols)
                plt.legend(loc='upper right')
                plt.savefig(
                    os.path.join(self.cfg.path.plot_file_path, f'{col}_distribution_by_{target_cols}_{epoch}.png'))

            for key in self.loss_keys:
                self.test_loss[key].append(total_losses[key] / len(self.dataloaders.test))


def trainer_main(cols: List[str] = None):
    config_manager.load_config()
    cfg = config_manager.config
    cfg.log.wandb.name = f'{cfg.log.time}_{cfg.log.ipaddr}'
    cfg.log.wandb.tags = ['TemporalAE']

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

    tc_cols = train_dataset.tc_cols
    tl_cols = train_dataset.tl_cols
    categorical_feature_info = []
    for c in (tc_cols + tl_cols):
        info = [info for info in train_dataset.temporal_transformer._data_manipulation_info_list if info.column_name == c][0]
        categorical_feature_info.append(info)
    categorical_feature_out_dims = [info.output_dimensions for info in categorical_feature_info]
    set_cfg(cfg, 'model.temporal_autoencoder.categorical_card', categorical_feature_out_dims)
    set_cfg(cfg, 'model.temporal_autoencoder.num_numerical', len(train_dataset.tn_cols))

    model = build_model(cfg.model.temporal_autoencoder,
                        device=torch.device(f'cuda:{cfg.device_num}'))
    if cfg.train.general.init_weight:
        initialize_weights(model)

    if cfg.train.general.distributed:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    loss_fn = set_loss_fn(cfg.train.loss)
    optimizer = set_optimizer(model.parameters(), cfg.train.optimizer, apply_lookahead=False)
    scheduler = set_scheduler(optimizer, cfg.train.scheduler)

    trainer = TemporalVAETrainer(config=cfg,
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

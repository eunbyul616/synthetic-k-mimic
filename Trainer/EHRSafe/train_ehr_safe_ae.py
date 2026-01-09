import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List

import torch

from Trainer.train import Trainer
from DataLoaders.loaderbase import get_dataloaders
from Utils.train import save_ckpt, set_loss_fn
from Utils.namespace import set_cfg
from Trainer.utils import *
from Loss.reconstruction import MSELoss, BCELoss
from Trainer.EHRSafe.train_gan import load_embedders
from Trainer.EHRSafe.utils import get_temporal_categorical_card, get_static_categorical_card


class EHRSafeAETrainer(Trainer):
    def __init__(self, config, rank, **kwargs):
        super(EHRSafeAETrainer, self).__init__(config=config, rank=rank, **kwargs)

        self.dataloaders = get_dataloaders(cfg=self.cfg,
                                           train_dataset=kwargs['train_dataset'],
                                           valid_dataset=kwargs['validation_dataset'],
                                           test_dataset=kwargs['test_dataset'],
                                           rank=self.rank,
                                           world_size=self.world_size,
                                           collate_fn=kwargs['collate_fn'])
        self.test_loss = {key: [] for key in self.loss_keys}
        self.logit_threshold = 0.5

        self.static_cate_ae = kwargs['static_cate_ae']
        self.temporal_cate_ae = kwargs['temporal_cate_ae']

        self.temporal_loss_w = 1.0
        self.static_loss_w = 1.0
        self.mask_loss_w = 1.0

        self.mse = MSELoss()
        self.bce = BCELoss()

        self.sc_cols = self.dataloaders.train.dataset.sc_cols
        self.tc_cols = self.dataloaders.train.dataset.tc_cols
        self.sl_cols = self.dataloaders.train.dataset.sl_cols
        self.tl_cols = self.dataloaders.train.dataset.tl_cols
        self.sn_cols = self.dataloaders.train.dataset.sn_cols
        self.tn_cols = self.dataloaders.train.dataset.tn_cols

        self.static_transformer = self.dataloaders.train.dataset.static_transformer
        self.temporal_transformer = self.dataloaders.train.dataset.temporal_transformer

        static_categorical_card, static_categorical_feature_info = get_static_categorical_card(self.dataloaders.train.dataset,
                                                                                        self.static_transformer)
        self.static_categorical_feature_info = static_categorical_feature_info
        self.sc_binary_cols = [col.column_name for col in self.static_categorical_feature_info if col.column_type == 'Binary']


        temporal_categorical_card, temporal_categorical_feature_info = get_temporal_categorical_card(self.dataloaders.train.dataset,
                                                                                            self.temporal_transformer)
        self.temporal_categorical_feature_info = temporal_categorical_feature_info
        self.tc_binary_cols = [col.column_name for col in self.temporal_categorical_feature_info if col.column_type == 'Binary']

        self.diagnoses_prefix = self.cfg.preprocess.icd_code.diagnoses_prefix
        self.procedure_prefix = self.cfg.preprocess.icd_code.procedure_prefix
        self.proc_prefix = self.cfg.preprocess.proc_prefix

    def _freeze_autoencoders(self):
        # freeze the weights of the static and temporal categorical autoencoders
        for param in self.static_cate_ae.parameters():
            param.requires_grad = False

        for param in self.temporal_cate_ae.parameters():
            param.requires_grad = False

    def run_epochs(self):
        self._freeze_autoencoders()

        for epoch in range(self.start_epoch, self.total_epochs+1):
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

    def _forward_batch(self, batch):
        sc, sn, sl, sm = self._unpack_static_batch(batch)
        tc, tn, tl, tm, _ = self._unpack_temporal_batch(batch)

        batch_size, seq_len, _ = tc.size()

        # seq dimension flatten
        tn_flat = tn.reshape(batch_size, -1)
        tm_flat = tm.reshape(batch_size, -1)

        # static categorical AE
        sc_in = torch.cat([sc, sl], dim=-1)
        sc_rep, _ = self.static_cate_ae(sc_in)

        # temporal categorical AE
        tc_in = torch.cat([tc, tl], dim=-1)
        tc_rep, _ = self.temporal_cate_ae(tc_in)

        # EHR-Safe AE input
        x = torch.cat([sc_rep, sn, tc_rep, tn_flat, sm, tm_flat], dim=-1)
        _, x_hat = self.model(x)

        s_idx = 0
        sc_rep_dim = sc_rep.shape[-1]
        sn_dim = sn.shape[-1]
        tc_rep_dim = tc_rep.shape[-1]
        tn_dim = tn_flat.shape[-1]
        sm_dim = sm.shape[-1]
        tm_dim = tm_flat.shape[-1]

        sc_rep_hat = x_hat[:, s_idx:s_idx + sc_rep_dim]
        s_idx += sc_rep_dim
        sn_hat = x_hat[:, s_idx:s_idx + sn_dim]
        s_idx += sn_dim

        tc_rep_hat = x_hat[:, s_idx:s_idx + tc_rep_dim]
        s_idx += tc_rep_dim
        tn_hat = x_hat[:, s_idx:s_idx + tn_dim]
        s_idx += tn_dim

        static_mask_hat = x_hat[:, s_idx:s_idx + sm_dim]
        s_idx += sm_dim
        temporal_mask_hat = x_hat[:, s_idx:s_idx + tm_dim]

        return {
            # shapes
            "batch_size": batch_size,
            "seq_len": seq_len,

            # original tensors
            "sc": sc,
            "sn": sn,
            "sl": sl,
            "sm": sm,
            "tc": tc,
            "tn": tn,
            "tl": tl,
            "tm": tm,
            "tn_flat": tn_flat,
            "tm_flat": tm_flat,

            # AE reps
            "sc_rep": sc_rep,
            "tc_rep": tc_rep,

            # recon from EHR-Safe AE
            "sc_rep_hat": sc_rep_hat,
            "sn_hat": sn_hat,
            "tc_rep_hat": tc_rep_hat,
            "tn_hat": tn_hat,
            "static_mask_hat": static_mask_hat,
            "temporal_mask_hat": temporal_mask_hat,
        }

    def _compute_losses(self, fwd_dict):
        pad_value = self.cfg.dataloader.pad_value

        sc_rep = fwd_dict["sc_rep"]
        sn = fwd_dict["sn"]
        tc_rep = fwd_dict["tc_rep"]
        tn_flat = fwd_dict["tn_flat"]
        sm = fwd_dict["sm"]
        tm_flat = fwd_dict["tm_flat"]

        sc_rep_hat = fwd_dict["sc_rep_hat"]
        sn_hat = fwd_dict["sn_hat"]
        tc_rep_hat = fwd_dict["tc_rep_hat"]
        tn_hat = fwd_dict["tn_hat"]
        static_mask_hat = fwd_dict["static_mask_hat"]
        temporal_mask_hat = fwd_dict["temporal_mask_hat"]

        # --- static loss ---
        s_loss_rep = self.mse(sc_rep_hat, sc_rep)
        # s_loss_num = self.mse(torch.sigmoid(sn_hat), sn)
        s_loss_num = self.mse(sn_hat, sn)
        s_loss = s_loss_rep + s_loss_num

        # --- temporal loss ---
        tn_mask = (tn_flat != pad_value)
        t_loss_rep = self.mse(tc_rep_hat, tc_rep)
        # t_loss_num = self.mse(torch.sigmoid(tn_hat), tn_flat, mask=tn_mask)
        t_loss_num = self.mse(tn_hat, tn_flat, mask=tn_mask)
        t_loss = t_loss_rep + t_loss_num

        # --- mask loss ---
        tm_mask = (tm_flat != pad_value)
        m_loss_static = self.bce(static_mask_hat, sm)
        m_loss_temporal = self.bce(temporal_mask_hat, tm_flat, mask=tm_mask)
        m_loss = m_loss_static + m_loss_temporal

        total_loss = (
                self.static_loss_w * s_loss
                + self.temporal_loss_w * t_loss
                + self.mask_loss_w * m_loss
        )

        batch_losses = {
            "Total_Loss": total_loss,
            "BCE_Loss": m_loss,
            "MSE_Loss": s_loss + t_loss,
            "Static_Loss": s_loss,
            "Temporal_Loss": t_loss,
            "Mask_Loss": m_loss,
        }

        return total_loss, batch_losses

    def _apply_static_activation(self, sc_hat, sn_hat, sm_hat, batch_size, logit_threshold=0.5):
        s_idx = 0
        act_sc_hat = []
        for i, col in enumerate(self.sc_cols + self.sl_cols):
            _x_hat = sc_hat[i]
            dim = _x_hat.shape[-1]

            if col in self.sc_cols:
                if col in self.sc_binary_cols:
                    _act_x_hat = torch.sigmoid(_x_hat)
                    _act_x_hat = (_act_x_hat >= logit_threshold).float()
                    act_sc_hat.append(_act_x_hat)

                else:
                    act_sc_hat.append(torch.softmax(_x_hat, dim=-1))

            elif col in self.sl_cols:
                _act_x_hat = torch.sigmoid(_x_hat)
                _act_x_hat = (_act_x_hat >= logit_threshold).float()
                act_sc_hat.append(_act_x_hat)

            s_idx += dim

        act_sc_hat = torch.cat(act_sc_hat, dim=-1)
        # act_sn_hat = torch.sigmoid(sn_hat)
        act_sn_hat = sn_hat
        act_sm_hat = torch.sigmoid(sm_hat)
        act_sm_hat = (act_sm_hat >= logit_threshold).float()

        return act_sc_hat, act_sn_hat, act_sm_hat

    def _apply_temporal_activation(self, tc_hat, tn_hat, tm_hat, batch_size, seq_len, logit_threshold=0.5):
        s_idx = 0
        act_x_hat = []
        for i, col in enumerate(self.tc_cols + self.tl_cols):
            _x_hat = tc_hat[i]
            _x_hat = _x_hat.view(batch_size, seq_len, -1)
            dim = _x_hat.shape[-1]

            if col in self.tc_cols:
                if col in self.tc_binary_cols:
                    _act_x_hat = torch.sigmoid(_x_hat)
                    _act_x_hat = (_act_x_hat >= self.logit_threshold).float()
                    act_x_hat.append(_act_x_hat)

                else:
                    act_x_hat.append(torch.softmax(_x_hat, dim=-1))

            elif col in self.tl_cols:
                _act_x_hat = torch.sigmoid(_x_hat)
                _act_x_hat = (_act_x_hat >= self.logit_threshold).float()
                act_x_hat.append(_act_x_hat)

            s_idx += dim
        act_tc_hat = torch.cat(act_x_hat, dim=-1)
        # act_tn_hat = torch.sigmoid(tn_hat)
        act_tn_hat = tn_hat
        act_tm_hat = torch.sigmoid(tm_hat)
        act_tm_hat = (act_tm_hat >= logit_threshold).float()

        return act_tc_hat, act_tn_hat, act_tm_hat


    def train_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        train_iterator = self._set_iterator(self.dataloaders.train, epoch, mode='Train')

        for batch in train_iterator:
            self.optimizer.zero_grad()

            out = self._forward_batch(batch)
            loss, batch_losses = self._compute_losses(out)

            loss.backward()
            self.optimizer.step()

            total_losses['Total_Loss'] += loss.item()
            total_losses['BCE_Loss'] += batch_losses['BCE_Loss'].item()
            total_losses['MSE_Loss'] += batch_losses['MSE_Loss'].item()
            total_losses['Static_Loss'] += batch_losses['Static_Loss'].item()
            total_losses['Temporal_Loss'] += batch_losses['Temporal_Loss'].item()
            total_losses['Mask_Loss'] += batch_losses['Mask_Loss'].item()

            self._set_iterator_postfix(train_iterator, total_losses['Total_Loss'])

        for key in self.loss_keys:
            self.train_loss[key].append(total_losses[key] / len(self.dataloaders.train))

    def validate_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}

        validation_iterator = self._set_iterator(self.dataloaders.valid, epoch, mode='val')

        with torch.no_grad():
            for batch in validation_iterator:
                out = self._forward_batch(batch)
                loss, batch_losses = self._compute_losses(out)

                total_losses['Total_Loss'] += loss.item()
                total_losses['BCE_Loss'] += batch_losses['BCE_Loss'].item()
                total_losses['MSE_Loss'] += batch_losses['MSE_Loss'].item()
                total_losses['Static_Loss'] += batch_losses['Static_Loss'].item()
                total_losses['Temporal_Loss'] += batch_losses['Temporal_Loss'].item()
                total_losses['Mask_Loss'] += batch_losses['Mask_Loss'].item()

                self._set_iterator_postfix(validation_iterator, total_losses['Total_Loss'])

            for key in self.loss_keys:
                self.validation_loss[key].append(total_losses[key] / len(self.dataloaders.valid))

    def eval_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        test_iterator = self._set_iterator(self.dataloaders.test, epoch, mode='test')

        if not os.path.exists(self.cfg.path.plot_file_path):
            os.makedirs(self.cfg.path.plot_file_path, exist_ok=True)

        static_transformer = self.dataloaders.test.dataset.static_transformer
        temporal_transformer = self.dataloaders.test.dataset.temporal_transformer

        with torch.no_grad():
            target = {'static_data': [], 'temporal_data': [], 'time_data': []}
            data_hat = {'sc_rep_hat': [], 'tc_rep_hat': [], 'sn_hat': [], 'tn_hat': [],
                        'static_mask_hat': [], 'temporal_mask_hat': [], 'time_hat': []}
            for batch in test_iterator:
                out = self._forward_batch(batch)
                loss, batch_losses = self._compute_losses(out)

                total_losses['Total_Loss'] += loss.item()
                total_losses['BCE_Loss'] += batch_losses['BCE_Loss'].item()
                total_losses['MSE_Loss'] += batch_losses['MSE_Loss'].item()
                total_losses['Static_Loss'] += batch_losses['Static_Loss'].item()
                total_losses['Temporal_Loss'] += batch_losses['Temporal_Loss'].item()
                total_losses['Mask_Loss'] += batch_losses['Mask_Loss'].item()
                self._set_iterator_postfix(test_iterator, total_losses['Total_Loss'])

                batch_size = out['batch_size']
                seq_len = out['seq_len']

            # ----- target data (real) -----
            static_x = torch.cat([out['sc'], out['sl'], out['sn'], out['sm']], dim=-1)
            target['static_data'].append(static_x)

            temporal_x = torch.cat([out['tc'], out['tl'], out['tn'], out['tm']], dim=-1)
            target['temporal_data'].append(temporal_x)

            # ----- reconstructed (latent & mask) -----
            data_hat['sc_rep_hat'].append(out['sc_rep_hat'])
            data_hat['tc_rep_hat'].append(out['tc_rep_hat'])
            data_hat['sn_hat'].append(out['sn_hat'])
            data_hat['static_mask_hat'].append(out['static_mask_hat'])
            data_hat['tn_hat'].append(
                out['tn_hat'].reshape(batch_size, seq_len, -1)
            )
            data_hat['temporal_mask_hat'].append(
                out['temporal_mask_hat'].reshape(batch_size, seq_len, -1)
            )

            # ----- concat over all batches -----
            target['static_data'] = torch.cat(target['static_data'], dim=0)
            target['temporal_data'] = torch.cat(target['temporal_data'], dim=0)

            data_hat['sc_rep_hat'] = torch.cat(data_hat['sc_rep_hat'], dim=0)
            data_hat['tc_rep_hat'] = torch.cat(data_hat['tc_rep_hat'], dim=0)
            data_hat['sn_hat'] = torch.cat(data_hat['sn_hat'], dim=0)
            data_hat['tn_hat'] = torch.cat(data_hat['tn_hat'], dim=0)
            data_hat['static_mask_hat'] = torch.cat(data_hat['static_mask_hat'], dim=0)
            data_hat['temporal_mask_hat'] = torch.cat(data_hat['temporal_mask_hat'], dim=0)

            # ----- feature info 구성 -----
            sc_feature_info = [
                info
                for info in static_transformer._data_manipulation_info_list
                if (info.column_type in ['Categorical', 'Binary', 'Listwise'])
                   and ('_mask' not in info.column_name)
            ]
            sn_feature_info = [
                info
                for info in static_transformer._data_manipulation_info_list
                if info.column_type == 'Numerical'
            ]

            sm_feature_info = []
            mask_info = [
                info
                for info in static_transformer._data_manipulation_info_list
                if '_mask' in info.column_name
            ]
            for s_info in (sc_feature_info + sn_feature_info):
                for info in mask_info:
                    if f'{s_info.column_name}_mask' == info.column_name:
                        sm_feature_info.append(info)
                        break

            tc_feature_info = [
                info
                for info in temporal_transformer._data_manipulation_info_list
                if (info.column_type in ['Categorical', 'Binary', 'Listwise'])
                   and ('_mask' not in info.column_name)
            ]
            tn_feature_info = [
                info
                for info in temporal_transformer._data_manipulation_info_list
                if info.column_type == 'Numerical'
            ]

            tm_feature_info = []
            mask_info = [
                info
                for info in temporal_transformer._data_manipulation_info_list
                if '_mask' in info.column_name
            ]
            for tc_info in (tc_feature_info + tn_feature_info):
                for info in mask_info:
                    if f'{tc_info.column_name}_mask' == info.column_name:
                        tm_feature_info.append(info)
                        break

            # ----- static reconstruction -----
            sc_hat = self.static_cate_ae.decoder(data_hat['sc_rep_hat'])
            act_sc_hat, act_sn_hat, act_sm_hat = self._apply_static_activation(sc_hat=sc_hat,
                                                                               sn_hat=data_hat['sn_hat'],
                                                                               sm_hat=data_hat['static_mask_hat'],
                                                                               batch_size=batch_size,
                                                                               logit_threshold=self.logit_threshold)
            static_data_hat = torch.cat([act_sc_hat, act_sn_hat, act_sm_hat], dim=-1)

            # ----- temporal reconstruction -----
            tc_hat = self.temporal_cate_ae.decoder(data_hat['tc_rep_hat'])
            act_tc_hat, act_tn_hat, act_tm_hat = self._apply_temporal_activation(tc_hat=tc_hat,
                                                                                 tn_hat=data_hat['tn_hat'],
                                                                                 tm_hat=data_hat['temporal_mask_hat'],
                                                                                 batch_size=batch_size,
                                                                                 seq_len=seq_len,
                                                                                 logit_threshold=self.logit_threshold)
            temporal_data_hat = torch.cat(
                [act_tc_hat, act_tn_hat, act_tm_hat], dim=-1
            )

            # ----- inverse transform -----
            static_feature_info = sc_feature_info + sn_feature_info + sm_feature_info
            static_data = target['static_data'].detach().cpu().numpy()
            static_data_hat = static_data_hat.detach().cpu().numpy()
            static_data, static_data_hat = inverse_transform(
                real=static_data,
                synthetic=static_data_hat,
                transformer=static_transformer,
                feature_info=static_feature_info,
            )

            mask = target['temporal_data'] != self.cfg.dataloader.pad_value
            mask = mask.reshape(-1, target['temporal_data'].shape[-1])
            mask = mask[:, 0].reshape(-1, 1).repeat(
                1, target['temporal_data'].shape[-1]
            )
            mask = mask.detach().cpu().numpy()

            temporal_feature_info = tc_feature_info + tn_feature_info + tm_feature_info
            temporal_data = (
                target['temporal_data']
                .reshape(-1, target['temporal_data'].shape[-1])
                .detach()
                .cpu()
                .numpy()
            )
            temporal_data_hat = (
                temporal_data_hat.reshape(-1, temporal_data_hat.shape[-1])
                .detach()
                .cpu()
                .numpy()
            )
            temporal_data, temporal_data_hat = inverse_transform(
                real=temporal_data,
                synthetic=temporal_data_hat,
                transformer=temporal_transformer,
                feature_info=temporal_feature_info,
                mask=mask,
            )

            # ----- column 리스트 -----
            sn_cols = self.dataloaders.test.dataset.sn_cols
            sc_cols = self.dataloaders.test.dataset.sc_cols
            tn_cols = self.dataloaders.test.dataset.tn_cols
            tc_cols = self.dataloaders.test.dataset.tc_cols

            # ----- evaluation -----
            # static
            eval_numerical_features(
                cols=sn_cols,
                real=static_data,
                synthetic=static_data_hat,
                labels=['Real', 'Reconstructed'],
                epoch=epoch,
                save_path=self.cfg.path.plot_file_path,
            )
            cols = (
                sc_cols
                + [
                    col
                    for col in static_data.columns
                    if (col.startswith(self.diagnoses_prefix)) and ('mask' not in col)
                ]
                + [
                    col
                    for col in static_data.columns
                    if (col.startswith(self.procedure_prefix)) and ('mask' not in col)
                ]
            )
            eval_categorical_features(
                cols=cols,
                real=static_data,
                synthetic=static_data_hat,
                labels=['Real', 'Reconstructed'],
                epoch=epoch,
                save_path=self.cfg.path.plot_file_path,
            )

            # temporal
            eval_numerical_features(
                cols=tn_cols,
                real=temporal_data,
                synthetic=temporal_data_hat,
                labels=['Real', 'Reconstructed'],
                epoch=epoch,
                save_path=self.cfg.path.plot_file_path,
            )

            cols = (
                tc_cols
                + [
                    col
                    for col in temporal_data.columns
                    if (col.startswith(self.proc_prefix)) and ('mask' not in col)
                ]
            )
            eval_categorical_features(
                cols=cols,
                real=temporal_data,
                synthetic=temporal_data_hat,
                labels=['Real', 'Reconstructed'],
                epoch=epoch,
                save_path=self.cfg.path.plot_file_path,
            )

            for key in self.loss_keys:
                self.test_loss[key].append(total_losses[key] / len(self.dataloaders.test))

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
        condition = None
        return tc_data, tn_data, tl_data, temporal_mask, condition


def load_embedders(cfg):
    from Utils.namespace import _load_yaml
    from Models.EHRSafe import StaticCategoricalAutoEncoder, TemporalCategoricalAutoEncoder

    checkpoint_saved_root = '/'.join(Path(cfg.path.ckpt_path).parts[:-1])
    # static categorical autoencoder
    static_cate_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                        cfg.train.static_categorical_ae.name,
                                                        cfg.train.static_categorical_ae.checkpoint)
    static_cate_config = _load_yaml(os.path.join(static_cate_embedder_checkpoint_path, 'config.yaml'))
    static_cate_ae = StaticCategoricalAutoEncoder.build_model(static_cate_config.model.static_categorical_autoencoder,
                                                              device=torch.device(f'cuda:{cfg.device_num}'))
    static_cate_ae_checkpoint = torch.load(os.path.join(static_cate_embedder_checkpoint_path, 'checkpoint_best.pth.tar'),
                                           map_location=f'cuda:{cfg.device_num}')
    static_cate_ae.load_state_dict(static_cate_ae_checkpoint['state_dict'])

    # temporal categorical autoencoder
    temporal_cate_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                          cfg.train.temporal_categorical_ae.name,
                                                          cfg.train.temporal_categorical_ae.checkpoint)
    temporal_cate_config = _load_yaml(os.path.join(temporal_cate_embedder_checkpoint_path, 'config.yaml'))
    temporal_cate_ae = TemporalCategoricalAutoEncoder.build_model(temporal_cate_config.model.temporal_categorical_autoencoder,
                                                                  device=torch.device(f'cuda:{cfg.device_num}'))
    temporal_cate_ae_checkpoint = torch.load(os.path.join(temporal_cate_embedder_checkpoint_path, 'checkpoint_best.pth.tar'),
                                             map_location=f'cuda:{cfg.device_num}')
    temporal_cate_ae.load_state_dict(temporal_cate_ae_checkpoint['state_dict'])

    return static_cate_ae, temporal_cate_ae


def ehr_safe_ae_trainer_main(cols):
    import config_manager

    from Models.EHRSafe.EHRSafeAutoEncoder import build_model
    from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset

    from Utils.reproducibility import lock_seed
    from Utils.train import set_loss_fn, set_scheduler, set_optimizer

    config_manager.load_config()
    cfg = config_manager.config
    cfg.log.wandb.name = f'{cfg.log.time}_{cfg.log.ipaddr}'
    cfg.log.wandb.tags = ['EHR-Safe AE']

    lock_seed(seed=cfg.seed, multi_gpu=False, activate_cudnn=False)

    dataset_name = cfg.dataset.dataset_name
    fname = cfg.preprocess.preprocess_fname_suffix
    train_ratio, test_ratio = cfg.preprocess.train_valid_test_split_ratio
    dataset_fname = f'{dataset_name}_{fname}_{int(train_ratio * 10)}.h5'

    train_dataset = CustomDataset(cfg=cfg,
                                  dataset_name=dataset_name,
                                  dataset_fname=dataset_fname,
                                  mode='train',
                                  condition_col=getattr(cfg.data, 'condition_col', None),
                                  static_cols=cols)
    validation_dataset = CustomDataset(cfg=cfg,
                                       dataset_name=dataset_name,
                                       dataset_fname=dataset_fname,
                                       mode='val',
                                       condition_col=getattr(cfg.data, 'condition_col', None),
                                       static_cols=cols)
    test_dataset = CustomDataset(cfg=cfg,
                                 dataset_name=dataset_name,
                                 dataset_fname=dataset_fname,
                                 mode='test',
                                 condition_col=getattr(cfg.data, 'condition_col', None),
                                 static_cols=cols)

    static_cate_ae, temporal_cate_ae = load_embedders(cfg)
    static_feature_dim = (static_cate_ae.encoder.embedding_dim +
                          sum(
                              [info.output_dimensions
                               for info in train_dataset.static_transformer._data_manipulation_info_list
                               if (info.column_type == 'Numerical') or ('mask' in info.column_name)]
                          ))
    temporal_feature_dim = temporal_cate_ae.encoder.embedding_dim + sum(
        [info.output_dimensions
         for info in train_dataset.temporal_transformer._data_manipulation_info_list
         if (info.column_type == 'Numerical') or ('mask' in info.column_name)]) * cfg.dataloader.seq_len
    feature_dim = static_feature_dim + temporal_feature_dim
    set_cfg(cfg, 'model.ehr_safe_autoencoder.encoder.input_dim', feature_dim)
    set_cfg(cfg, 'model.ehr_safe_autoencoder.decoder.output_dim', feature_dim)

    # build ehr-safe autoencoder
    model = build_model(cfg.model.ehr_safe_autoencoder, device=torch.device(f'cuda:{cfg.device_num}'))
    if cfg.train.general.init_weight:
        model.apply(model.init_weights)

    if cfg.train.general.distributed:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    loss_fn = set_loss_fn(cfg.train.loss)
    optimizer = set_optimizer(model.parameters(), cfg.train.optimizer, apply_lookahead=cfg.train.optimizer.lookahead.flag)
    scheduler = set_scheduler(optimizer, cfg.train.scheduler)

    trainer = EHRSafeAETrainer(config=cfg,
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
                               static_cate_ae=static_cate_ae,
                               temporal_cate_ae=temporal_cate_ae)
    trainer.run_epochs()


if __name__ == '__main__':
    cols = None
    ehr_safe_ae_trainer_main(cols)
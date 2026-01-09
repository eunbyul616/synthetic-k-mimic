import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch

from Trainer.train import Trainer
from Trainer.utils import *

from DataLoaders.loaderbase import get_dataloaders
from Utils.namespace import set_cfg
from Utils.train import save_ckpt, load_ckpt

from Visualization import *
from Trainer.EHRSafe.utils import get_temporal_categorical_card, get_static_categorical_card


class GANTrainer(Trainer):
    def __init__(self, config, rank, **kwargs):
        super(GANTrainer, self).__init__(config=config, rank=rank, **kwargs)

        self.static_cate_ae = kwargs['static_cate_ae']
        self.temporal_cate_ae = kwargs['temporal_cate_ae']
        self.ehr_safe_ae = kwargs['ehr_safe_ae']

        self.disc_loss_fn = kwargs['loss_fn']
        self.gen_loss_fn = kwargs['gen_loss_fn']
        self.disc_optimizer = kwargs['optimizer']
        self.gen_optimizer = kwargs['gen_optimizer']
        self.disc_scheduler = kwargs['scheduler']
        self.gen_scheduler = kwargs['gen_scheduler']

        self.start_epoch = load_ckpt(
            self.cfg,
            self.model,
            self.rank,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
            gen_scheduler=self.gen_scheduler,
            disc_scheduler=self.disc_scheduler)

        self.dataloaders = get_dataloaders(cfg=self.cfg,
                                           train_dataset=kwargs['train_dataset'],
                                           valid_dataset=kwargs['validation_dataset'],
                                           test_dataset=kwargs['test_dataset'],
                                           rank=self.rank,
                                           world_size=self.world_size,
                                           collate_fn=kwargs['collate_fn'])
        self.test_loss = {key: [] for key in self.loss_keys}
        self.logit_threshold = 0.5

        self.sc_cols = self.dataloaders.train.dataset.sc_cols
        self.sl_cols = self.dataloaders.train.dataset.sl_cols
        self.tc_cols = self.dataloaders.train.dataset.tc_cols
        self.tl_cols = self.dataloaders.train.dataset.tl_cols
        self.sn_cols = self.dataloaders.train.dataset.sn_cols
        self.tn_cols = self.dataloaders.train.dataset.tn_cols

        self.static_transformer = self.dataloaders.train.dataset.static_transformer
        self.temporal_transformer = self.dataloaders.train.dataset.temporal_transformer

        static_categorical_card, static_categorical_feature_info = get_static_categorical_card(
            self.dataloaders.train.dataset,
            self.static_transformer)
        self.static_categorical_feature_info = static_categorical_feature_info
        self.sc_binary_cols = [col.column_name for col in self.static_categorical_feature_info if
                               col.column_type == 'Binary']

        temporal_categorical_card, temporal_categorical_feature_info = get_temporal_categorical_card(
            self.dataloaders.train.dataset,
            self.temporal_transformer)
        self.temporal_categorical_feature_info = temporal_categorical_feature_info
        self.tc_binary_cols = [col.column_name for col in self.temporal_categorical_feature_info if
                               col.column_type == 'Binary']

        self.diagnoses_prefix = self.cfg.preprocess.icd_code.diagnoses_prefix
        self.procedure_prefix = self.cfg.preprocess.icd_code.procedure_prefix
        self.proc_prefix = self.cfg.preprocess.proc_prefix

    def _freeze_autoencoders(self):
        # freeze the weights of the static and temporal categorical autoencoders
        for param in self.static_cate_ae.parameters():
            param.requires_grad = False
        for param in self.temporal_cate_ae.parameters():
            param.requires_grad = False
        for param in self.ehr_safe_ae.parameters():
            param.requires_grad = False

    def _set_iterator_postfix(self, iterator, loss, disc_loss, gen_loss):
        iterator.set_postfix(loss=loss / (iterator.n + 1),
                             disc_loss=disc_loss / (iterator.n + 1),
                             gen_loss=gen_loss / (iterator.n + 1),
                             disc_lr=self.disc_optimizer.param_groups[0]['lr'],
                             gen_lr=self.gen_optimizer.param_groups[0]['lr'])

    def _forward_batch(self, batch):
        # Encode a batch into EHR-Safe latent rep using:
        # StaticCategoricalAE + TemporalCategoricalAE + EHRSafeAutoEncoder.
        sc, sn, sl, sm = self._unpack_static_batch(batch)
        tc, tn, tl, tm, _ = self._unpack_temporal_batch(batch)

        # concat list-wise categorical
        sc_in = torch.cat([sc, sl], dim=-1)
        tc_in = torch.cat([tc, tl], dim=-1)

        batch_size, seq_len, _ = tc_in.size()

        # flatten temporal numerical & masks across time
        tn_flat = tn.reshape(batch_size, -1)
        tm_flat = tm.reshape(batch_size, -1)

        # encoding
        sc_rep, _ = self.static_cate_ae(sc_in)
        tc_rep, _ = self.temporal_cate_ae(tc_in)

        x = torch.cat([sc_rep, sn, tc_rep, tn_flat, sm, tm_flat], dim=-1)
        rep, _ = self.ehr_safe_ae(x)

        return {
            "rep": rep,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "sc": sc,
            "sn": sn,
            "sl": sl,
            "sm": sm,
            "tc": tc,
            "tn": tn,
            "tl": tl,
            "tm": tm,
            "sc_rep": sc_rep,
            "tc_rep": tc_rep,
        }

    def get_feature_info(self):
        sc_feature_info = [
            info
            for info in self.static_transformer._data_manipulation_info_list
            if (info.column_type in ['Categorical', 'Binary', 'Listwise'])
               and ('_mask' not in info.column_name)
        ]
        sn_feature_info = [
            info
            for info in self.static_transformer._data_manipulation_info_list
            if info.column_type == 'Numerical'
        ]

        sm_feature_info = []
        mask_info = [
            info
            for info in self.static_transformer._data_manipulation_info_list
            if '_mask' in info.column_name
        ]
        for s_info in (sc_feature_info + sn_feature_info):
            for info in mask_info:
                if f'{s_info.column_name}_mask' == info.column_name:
                    sm_feature_info.append(info)
                    break

        tc_feature_info = [
            info
            for info in self.temporal_transformer._data_manipulation_info_list
            if (info.column_type in ['Categorical', 'Binary', 'Listwise'])
               and ('_mask' not in info.column_name)
        ]
        tn_feature_info = [
            info
            for info in self.temporal_transformer._data_manipulation_info_list
            if info.column_type == 'Numerical'
        ]

        tm_feature_info = []
        mask_info = [
            info
            for info in self.temporal_transformer._data_manipulation_info_list
            if '_mask' in info.column_name
        ]
        for tc_info in (tc_feature_info + tn_feature_info):
            for info in mask_info:
                if f'{tc_info.column_name}_mask' == info.column_name:
                    tm_feature_info.append(info)
                    break

        return sc_feature_info, sn_feature_info, sm_feature_info, tc_feature_info, tn_feature_info, tm_feature_info

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
        act_tn_hat = tn_hat.view(batch_size, seq_len, -1)
        act_tm_hat = torch.sigmoid(tm_hat)
        act_tm_hat = (act_tm_hat >= logit_threshold).float()
        act_tm_hat = act_tm_hat.view(batch_size, seq_len, -1)

        return act_tc_hat, act_tn_hat, act_tm_hat

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
                validation_loss=self.validation_loss['Gen_Loss'],
                states={'epoch': epoch,
                        'arch': self.cfg.log.time,
                        'model': self.model,
                        'optimizer': self.disc_optimizer.state_dict(),
                        'scheduler': self.disc_scheduler.state_dict() if self.disc_scheduler is not None else None,
                        'gen_optimizer': self.gen_optimizer.state_dict(),
                        'gen_scheduler': self.gen_scheduler.state_dict() if self.gen_scheduler is not None else None
                        },
                rank=self.rank,
                save_condition='lower'
            )

    def train_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}

        train_iterator = self._set_iterator(self.dataloaders.train, epoch, mode='Train')

        for batch in train_iterator:
            out = self._forward_batch(batch)
            rep = out['rep']

            self.model.discriminator.train()
            self.model.generator.eval()
            for _ in range(self.cfg.train.discriminator_steps):
                self.disc_optimizer.zero_grad()

                disc_loss = 0
                gen_out = self.model(rep)
                fake = gen_out['fake']
                disc_fake = gen_out['disc_fake']
                disc_real = gen_out['disc_real']
                gp = self.model.calculate_gradient_penalty(rep, fake, self.device)
                disc_loss = self.disc_loss_fn(disc_fake, disc_real)

                gp.backward(retain_graph=True)
                disc_loss.backward()
                self.disc_optimizer.step()

            self.model.discriminator.eval()
            self.model.generator.train()

            self.gen_optimizer.zero_grad()
            gen_loss = 0
            gen_out = self.model(rep)
            fake = gen_out['fake']
            disc_fake = gen_out['disc_fake']
            disc_real = gen_out['disc_real']
            gen_loss = self.gen_loss_fn(disc_fake, disc_real)
            gen_loss.backward()
            self.gen_optimizer.step()

            loss = disc_loss + gen_loss
            total_losses['Total_Loss'] += loss.item()
            total_losses['Disc_Loss'] += disc_loss.item()
            total_losses['Gen_Loss'] += gen_loss.item()

            self._set_iterator_postfix(train_iterator, total_losses['Total_Loss'], total_losses['Disc_Loss'], total_losses['Gen_Loss'])

        for key in self.loss_keys:
            self.train_loss[key].append(total_losses[key] / len(self.dataloaders.train))

    def validate_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}

        validation_iterator = self._set_iterator(self.dataloaders.valid, epoch, mode='val')

        with torch.no_grad():
            for batch in validation_iterator:
                out = self._forward_batch(batch)
                rep = out['rep']

                disc_loss = 0
                gen_loss = 0
                loss = 0

                gen_out = self.model(rep)
                fake = gen_out['fake']
                disc_fake = gen_out['disc_fake']
                disc_real = gen_out['disc_real']

                disc_loss = self.disc_loss_fn(disc_fake, disc_real)
                gen_loss = self.gen_loss_fn(disc_fake, disc_real)
                loss = disc_loss + gen_loss

                total_losses['Total_Loss'] += loss.item()
                total_losses['Disc_Loss'] += disc_loss.item()
                total_losses['Gen_Loss'] += gen_loss.item()
                self._set_iterator_postfix(validation_iterator, total_losses['Total_Loss'], total_losses['Disc_Loss'], total_losses['Gen_Loss'])

            for key in self.loss_keys:
                self.validation_loss[key].append(total_losses[key] / len(self.dataloaders.valid))

    def eval_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        test_iterator = self._set_iterator(self.dataloaders.test, epoch, mode='test')

        if not os.path.exists(self.cfg.path.plot_file_path):
            os.makedirs(self.cfg.path.plot_file_path, exist_ok=True)

        static_transformer = self.dataloaders.test.dataset.static_transformer
        temporal_transformer = self.dataloaders.test.dataset.temporal_transformer

        target = {'static_data': [], 'temporal_data': [], 'time_data': []}
        data_hat = {'sc_rep_hat': [], 'tc_rep_hat': [], 'sn_hat': [], 'tn_hat': [],
                    'static_mask_hat': [], 'temporal_mask_hat': [], 'time_hat': []}

        (
            sc_feature_info,
            sn_feature_info,
            sm_feature_info,
            tc_feature_info,
            tn_feature_info,
            tm_feature_info,
        ) = self.get_feature_info()

        with torch.no_grad():
            for batch in test_iterator:
                out = self._forward_batch(batch)
                rep = out['rep']
                batch_size = out['batch_size']
                seq_len = out['seq_len']

                # ----- target data (real) -----
                static_x = torch.cat([out['sc'], out['sl'], out['sn'], out['sm']], dim=-1)
                target['static_data'].append(static_x)

                temporal_x = torch.cat([out['tc'], out['tl'], out['tn'], out['tm']], dim=-1)
                target['temporal_data'].append(temporal_x)

                # GAN forward
                gan_out = self.model(rep)
                fake = gan_out['fake']
                disc_fake = gan_out['disc_fake']
                disc_real = gan_out['disc_real']

                disc_loss = self.disc_loss_fn(disc_fake, disc_real)
                gen_loss = self.gen_loss_fn(disc_fake, disc_real)
                loss = disc_loss + gen_loss

                total_losses['Total_Loss'] += loss.item()
                total_losses['Disc_Loss'] += disc_loss.item()
                total_losses['Gen_Loss'] += gen_loss.item()
                self._set_iterator_postfix(
                    test_iterator,
                    total_losses['Total_Loss'],
                    total_losses['Disc_Loss'],
                    total_losses['Gen_Loss'],
                )

                # decode fake latent with EHR-Safe decoder
                dec_fake = self.ehr_safe_ae.decoder(fake)

                sc_rep_dim = self.cfg.model.static_categorical_autoencoder.decoder.embedding_dim
                tc_rep_dim = self.cfg.model.temporal_categorical_autoencoder.decoder.embedding_dim
                sn_dim = sum([info.output_dimensions for info in sn_feature_info])
                tn_dim = sum([info.output_dimensions for info in tn_feature_info])
                sm_dim = sum([info.output_dimensions for info in sm_feature_info])
                tm_dim = sum([info.output_dimensions for info in tm_feature_info])
                (
                    sc_rep_hat,
                    tc_rep_hat,
                    sn_hat,
                    tn_hat,
                    static_mask_hat,
                    temporal_mask_hat
                ) = self._split_data_by_type(dec_fake, sc_rep_dim, tc_rep_dim, sn_dim, tn_dim, sm_dim, tm_dim, seq_len)

                data_hat['sc_rep_hat'].append(sc_rep_hat)
                data_hat['tc_rep_hat'].append(tc_rep_hat)
                data_hat['sn_hat'].append(sn_hat)
                data_hat['static_mask_hat'].append(static_mask_hat)
                data_hat['tn_hat'].append(tn_hat.view(batch_size, seq_len, -1))
                data_hat['temporal_mask_hat'].append(
                    temporal_mask_hat.view(batch_size, seq_len, -1)
                )

            # concat over all batches
            target['static_data'] = torch.cat(target['static_data'], dim=0)
            target['temporal_data'] = torch.cat(target['temporal_data'], dim=0)

            data_hat['sc_rep_hat'] = torch.cat(data_hat['sc_rep_hat'], dim=0)
            data_hat['tc_rep_hat'] = torch.cat(data_hat['tc_rep_hat'], dim=0)
            data_hat['sn_hat'] = torch.cat(data_hat['sn_hat'], dim=0)
            data_hat['tn_hat'] = torch.cat(data_hat['tn_hat'], dim=0)
            data_hat['static_mask_hat'] = torch.cat(data_hat['static_mask_hat'], dim=0)
            data_hat['temporal_mask_hat'] = torch.cat(
                data_hat['temporal_mask_hat'], dim=0
            )

            # ----- static reconstruction -----
            sc_hat = self.static_cate_ae.decoder(data_hat['sc_rep_hat'])
            act_sc_hat, act_sn_hat, act_sm_hat = self._apply_static_activation(sc_hat=sc_hat,
                                                                               sn_hat=data_hat['sn_hat'],
                                                                               sm_hat=data_hat['static_mask_hat'],
                                                                               batch_size=data_hat['sn_hat'].shape[0],
                                                                               logit_threshold=self.logit_threshold)
            static_data_hat = torch.cat([act_sc_hat, act_sn_hat, act_sm_hat], dim=-1)

            # ----- temporal reconstruction -----
            tc_hat = self.temporal_cate_ae.decoder(data_hat['tc_rep_hat'])
            act_tc_hat, act_tn_hat, act_tm_hat = self._apply_temporal_activation(tc_hat=tc_hat,
                                                                                 tn_hat=data_hat['tn_hat'],
                                                                                 tm_hat=data_hat['temporal_mask_hat'],
                                                                                 batch_size=data_hat['tn_hat'].shape[0],
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

        sn_cols = self.dataloaders.test.dataset.sn_cols
        sc_cols = self.dataloaders.test.dataset.sc_cols
        tn_cols = self.dataloaders.test.dataset.tn_cols
        tc_cols = self.dataloaders.test.dataset.tc_cols

        # ----- evaluation -----
        # static features
        eval_numerical_features(
            cols=sn_cols,
            real=static_data,
            synthetic=static_data_hat,
            labels=['Real', 'Synthetic'],
            epoch=epoch,
            save_path=self.cfg.path.plot_file_path,
        )
        cols = (
                sc_cols
                + [
                    col
                    for col in static_data.columns
                    if (self.diagnoses_prefix in col) and ('mask' not in col)
                ]
                + [
                    col
                    for col in static_data.columns
                    if (self.procedure_prefix in col) and ('mask' not in col)
                ]
        )
        eval_categorical_features(
            cols=cols,
            real=static_data,
            synthetic=static_data_hat,
            labels=['Real', 'Synthetic'],
            epoch=epoch,
            save_path=self.cfg.path.plot_file_path,
        )

        # temporal features
        eval_numerical_features(
            cols=tn_cols,
            real=temporal_data,
            synthetic=temporal_data_hat,
            labels=['Real', 'Synthetic'],
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

    def _split_data_by_type(self, x_hat, sc_rep_dim, tc_rep_dim, sn_dim, tn_dim, sm_dim, tm_dim, seq_len):
        s_idx = 0
        # static features
        rep_dim = sc_rep_dim
        dim = sc_rep_dim + sn_dim
        sc_rep_hat = x_hat[:, s_idx:s_idx + rep_dim]
        sn_hat = x_hat[:, s_idx + rep_dim:s_idx + dim]
        s_idx += dim

        # temporal features
        rep_dim = tc_rep_dim
        dim = tc_rep_dim + (seq_len * tn_dim)
        tc_rep_hat = x_hat[:, s_idx:s_idx + rep_dim]
        tn_hat = x_hat[:, s_idx + rep_dim:s_idx + dim]
        s_idx += dim

        # mask loss
        sm_dim = sm_dim
        dim = sm_dim + (seq_len * tm_dim)
        static_mask_hat = x_hat[:, s_idx:s_idx + sm_dim]
        temporal_mask_hat = x_hat[:, s_idx + sm_dim:s_idx + dim]
        s_idx += dim

        return sc_rep_hat, tc_rep_hat, sn_hat, tn_hat, static_mask_hat, temporal_mask_hat

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
    from Models.EHRSafe import StaticCategoricalAutoEncoder, TemporalCategoricalAutoEncoder, EHRSafeAutoEncoder

    # load embedders
    checkpoint_saved_root = '/'.join(Path(cfg.path.ckpt_path).parts[:-1])

    # static categorical autoencoder
    static_cate_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                        cfg.train.static_categorical_ae.name,
                                                        cfg.train.static_categorical_ae.checkpoint)
    static_cate_config = _load_yaml(os.path.join(static_cate_embedder_checkpoint_path, 'config.yaml'))
    static_cate_ae = StaticCategoricalAutoEncoder.build_model(static_cate_config.model.static_categorical_autoencoder,
                                                              device=torch.device(f'cuda:{cfg.device_num}'))
    static_cate_ae_checkpoint = torch.load(
        os.path.join(static_cate_embedder_checkpoint_path, 'checkpoint_best.pth.tar'),
        map_location=f'cuda:{cfg.device_num}')
    static_cate_ae.load_state_dict(static_cate_ae_checkpoint['state_dict'])

    # temporal categorical autoencoder
    temporal_cate_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                          cfg.train.temporal_categorical_ae.name,
                                                          cfg.train.temporal_categorical_ae.checkpoint)
    temporal_cate_config = _load_yaml(os.path.join(temporal_cate_embedder_checkpoint_path, 'config.yaml'))
    temporal_cate_ae = TemporalCategoricalAutoEncoder.build_model(
        temporal_cate_config.model.temporal_categorical_autoencoder,
        device=torch.device(f'cuda:{cfg.device_num}'))
    temporal_cate_ae_checkpoint = torch.load(
        os.path.join(temporal_cate_embedder_checkpoint_path, 'checkpoint_best.pth.tar'),
        map_location=f'cuda:{cfg.device_num}')
    temporal_cate_ae.load_state_dict(temporal_cate_ae_checkpoint['state_dict'])

    # ehr safe autoencoder
    ehr_safe_embedder_checkpoint_path = os.path.join(checkpoint_saved_root,
                                                     cfg.train.ehr_safe_ae.name,
                                                     cfg.train.ehr_safe_ae.checkpoint)
    ehr_safe_config = _load_yaml(os.path.join(ehr_safe_embedder_checkpoint_path, 'config.yaml'))
    ehr_safe_ae = EHRSafeAutoEncoder.build_model(
        ehr_safe_config.model.ehr_safe_autoencoder,
        device=torch.device(f'cuda:{cfg.device_num}'))
    ehr_safe_ae_checkpoint = torch.load(
        os.path.join(ehr_safe_embedder_checkpoint_path, 'checkpoint_best.pth.tar'),
        map_location=f'cuda:{cfg.device_num}')
    ehr_safe_ae.load_state_dict(ehr_safe_ae_checkpoint['state_dict'])

    return static_cate_ae, temporal_cate_ae, ehr_safe_ae

def gan_trainer_main(cols):
    import config_manager

    from Models.EHR_Safe.GAN import build_model
    from Models.EHR_Safe import StaticCategoricalAutoEncoder, TemporalCategoricalAutoEncoder, EHRSafeAutoEncoder
    from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset

    from Utils.reproducibility import lock_seed
    from Utils.train import set_loss_fn, set_scheduler, set_optimizer

    config_manager.load_config()
    cfg = config_manager.config
    cfg.log.wandb.name = f'{cfg.log.time}_{cfg.log.ipaddr}'
    cfg.log.wandb.tags = ['GAN']

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

    # load embedders
    static_cate_ae, temporal_cate_ae, ehr_safe_ae = load_embedders(cfg)

    # update model config following the dataset
    set_cfg(cfg, 'model.gan.discriminator.input_dim', ehr_safe_ae.encoder.embedding_dim)
    model = build_model(cfg.model.gan, device=torch.device(f'cuda:{cfg.device_num}'))
    if cfg.train.general.init_weight:
        model.apply(model.init_weights)

    if cfg.train.general.distributed:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    disc_loss_fn = set_loss_fn(cfg.train.loss)
    gen_loss_fn = set_loss_fn(cfg.train.gen_loss)
    disc_optimizer = set_optimizer(model.discriminator.parameters(), cfg.train.optimizer,
                                   apply_lookahead=cfg.train.optimizer.lookahead.flag)
    gen_optimizer = set_optimizer(model.generator.parameters(), cfg.train.optimizer,
                                  apply_lookahead=cfg.train.optimizer.lookahead.flag)
    disc_scheduler = set_scheduler(disc_optimizer, cfg.train.scheduler)
    gen_scheduler = set_scheduler(gen_optimizer, cfg.train.scheduler)

    trainer = GANTrainer(config=cfg,
                         rank=cfg.device_num,
                         world_size=world_size,
                         model=model,
                         collate_fn=None,
                         loss_fn=disc_loss_fn,
                         optimizer=disc_optimizer,
                         scheduler=disc_scheduler,
                         gen_loss_fn=gen_loss_fn,
                         gen_optimizer=gen_optimizer,
                         gen_scheduler=gen_scheduler,
                         train_dataset=train_dataset,
                         validation_dataset=validation_dataset,
                         test_dataset=test_dataset,
                         static_cate_ae=static_cate_ae,
                         temporal_cate_ae=temporal_cate_ae,
                         ehr_safe_ae=ehr_safe_ae)
    trainer.run_epochs()


if __name__ == '__main__':
    cols = None
    gan_trainer_main(cols)
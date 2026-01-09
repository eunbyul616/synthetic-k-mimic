import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

import torch
from torch.utils.data import WeightedRandomSampler

from Trainer.train import Trainer
from DataLoaders.loaderbase import get_dataloaders
from Utils.train import save_ckpt, initialize_weights
from Utils.namespace import set_cfg
from Visualization import *
from Trainer.EHRSafe.utils import get_static_categorical_card


class StaticCategoricalAETrainer(Trainer):
    def __init__(self, config, rank, **kwargs):
        super(StaticCategoricalAETrainer, self).__init__(config=config, rank=rank, **kwargs)

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
        self.logit_threshold = 0.5

        self.transformer = self.dataloaders.train.dataset.static_transformer

        self.sc_cols = self.dataloaders.train.dataset.sc_cols
        self.sl_cols = self.dataloaders.train.dataset.sl_cols

        static_categorical_card, categorical_feature_info = get_static_categorical_card(self.dataloaders.train.dataset,
                                                                                        self.transformer)
        self.categorical_feature_info = categorical_feature_info

        self.sc_binary_cols = [col.column_name for col in self.categorical_feature_info if col.column_type == 'Binary']

        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.diagnoses_prefix = self.cfg.preprocess.icd_code.diagnoses_prefix
        self.procedure_prefix = self.cfg.preprocess.icd_code.procedure_prefix
        self.proc_prefix = self.cfg.preprocess.proc_prefix

    def run_epochs(self):
        for epoch in range(self.start_epoch, self.total_epochs+1):
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

            sc_data, sn_data, sl_data, static_mask, condition = self._unpack_batch(batch)
            data = torch.cat([sc_data, sl_data], dim=-1)
            rep, x_hat = self.model(data)

            loss = 0
            s_idx = 0
            for i, col in enumerate(self.sc_cols + self.sl_cols):
                dim = x_hat[i].shape[-1]
                # categorical feature
                if col in self.sc_cols:
                    if col in self.sc_binary_cols:
                        loss = loss + self.bce_loss_fn(x_hat[i], data[:, s_idx:s_idx+dim]).mean()
                    else:
                        loss = loss + self.ce_loss_fn(x_hat[i], data[:, s_idx:s_idx+dim]).mean()

                # listwise feature
                elif col in self.sl_cols:
                    loss = loss + self.bce_loss_fn(x_hat[i], data[:, s_idx:s_idx+dim]).mean()

                s_idx += dim

            total_losses['Total_Loss'] += loss.item()
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
                data = torch.cat([sc_data, sl_data], dim=-1)
                batch_size, feature_dim = data.size()
                rep, x_hat = self.model(data)

                loss = 0
                s_idx = 0
                for i, col in enumerate(self.sc_cols + self.sl_cols):
                    dim = x_hat[i].shape[-1]
                    # categorical feature
                    if col in self.sc_cols:
                        if col in self.sc_binary_cols:
                            loss = loss + self.bce_loss_fn(x_hat[i], data[:, s_idx:s_idx + dim]).mean()
                        else:
                            loss = loss + self.ce_loss_fn(x_hat[i], data[:, s_idx:s_idx + dim]).mean()
                    # listwise feature
                    elif col in self.sl_cols:
                        loss = loss + self.bce_loss_fn(x_hat[i], data[:, s_idx:s_idx + dim]).mean()

                    s_idx += dim

                total_losses['Total_Loss'] += loss.item()
                self._set_iterator_postfix(validation_iterator, total_losses['Total_Loss'])

            for key in self.loss_keys:
                self.validation_loss[key].append(total_losses[key] / len(self.dataloaders.valid))

    def eval_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        test_iterator = self._set_iterator(self.dataloaders.test, epoch, mode='test')

        transformer = self.dataloaders.test.dataset.static_transformer

        if not os.path.exists(self.cfg.path.plot_file_path):
            os.makedirs(self.cfg.path.plot_file_path, exist_ok=True)

        with torch.no_grad():
            target = []
            data_hat = []
            for batch in test_iterator:
                sc_data, sn_data, sl_data, static_mask, condition = self._unpack_batch(batch)
                data = torch.cat([sc_data, sl_data], dim=-1)
                batch_size, feature_dim = data.size()
                rep, x_hat = self.model(data)

                loss = 0
                s_idx = 0
                act_x_hat = []
                for i, col in enumerate(self.sc_cols + self.sl_cols):
                    dim = x_hat[i].shape[-1]
                    # categorical feature
                    if col in self.sc_cols:
                        if col in self.sc_binary_cols:
                            loss = loss + self.bce_loss_fn(x_hat[i], data[:, s_idx:s_idx + dim]).mean()
                        else:
                            loss = loss + self.ce_loss_fn(x_hat[i], data[:, s_idx:s_idx + dim]).mean()

                        if self.categorical_feature_info[i].column_type == 'Binary':
                            _act_x_hat = torch.sigmoid(x_hat[i])
                            _act_x_hat = (_act_x_hat >= self.logit_threshold).float()
                            act_x_hat.append(_act_x_hat)
                        elif self.categorical_feature_info[i].column_type == 'Categorical':
                            act_x_hat.append(torch.softmax(x_hat[i], dim=-1))

                    # listwise feature
                    elif col in self.sl_cols:
                        loss = loss + self.bce_loss_fn(x_hat[i], data[:, s_idx:s_idx + dim]).mean()

                        _act_x_hat = torch.sigmoid(x_hat[i])
                        _act_x_hat = (_act_x_hat >= self.logit_threshold).float()
                        act_x_hat.append(_act_x_hat)

                    s_idx += dim

                total_losses['Total_Loss'] += loss.item()
                self._set_iterator_postfix(test_iterator, total_losses['Total_Loss'])

                act_x_hat = torch.concatenate(act_x_hat, dim=-1)
                # x_hat = torch.concatenate(x_hat, dim=-1)
                target.append(data)
                data_hat.append(act_x_hat)
            target = torch.concatenate(target, dim=0)
            data_hat = torch.concatenate(data_hat, dim=0)

            target = transformer.inverse_transform(target.detach().cpu().numpy(), self.categorical_feature_info)
            data_hat = transformer.inverse_transform(data_hat.detach().cpu().numpy(), self.categorical_feature_info)

            for col in target.columns:
                countplot_categorical_feature(data1=target,
                                              data2=data_hat,
                                              col=col,
                                              stat='percent',
                                              label1='Real',
                                              label2='Reconstructed',
                                              title=col,
                                              save_path=self.cfg.path.plot_file_path + f'countplot_{col}_epoch_{epoch}.png')

            for key in self.loss_keys:
                self.test_loss[key].append(total_losses[key] / len(self.dataloaders.test))

    def _unpack_batch(self, batch):
        sc_data = batch[0].to(self.device)
        sn_data = batch[2].to(self.device)
        sl_data = batch[4].to(self.device)
        static_mask = batch[6].to(self.device)
        condition = None
        return sc_data, sn_data, sl_data, static_mask, condition


def static_categorical_ae_trainer_main(cols: List[str]=None):
    import config_manager

    from Models.EHRSafe.StaticCategoricalAutoEncoder import build_model
    from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset

    from Utils.reproducibility import lock_seed
    from Utils.train import set_loss_fn, set_scheduler, set_optimizer

    config_manager.load_config()
    cfg = config_manager.config
    cfg.log.wandb.name = f'{cfg.log.time}_{cfg.log.ipaddr}'
    cfg.log.wandb.tags = ['StaticCategoricalAE']

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
                                  # condition_col=cfg.data.condition_col,
                                  static_cols=cols)
    validation_dataset = CustomDataset(cfg=cfg,
                                       dataset_name=dataset_name,
                                       dataset_fname=dataset_fname,
                                       mode='val',
                                       condition_col=getattr(cfg.data, 'condition_col', None),
                                       # condition_col=cfg.data.condition_col,
                                       static_cols=cols)
    test_dataset = CustomDataset(cfg=cfg,
                                 dataset_name=dataset_name,
                                 dataset_fname=dataset_fname,
                                 mode='test',
                                 condition_col=getattr(cfg.data, 'condition_col', None),
                                 # condition_col=cfg.data.condition_col,
                                 static_cols=cols)

    train_sampler = None
    validation_sampler = None

    # update model config following the dataset
    categorical_feature_out_dims, categorical_feature_info = get_static_categorical_card(train_dataset,
                                                                                         train_dataset.static_transformer)

    set_cfg(cfg, 'model.static_categorical_autoencoder.encoder.input_dim', sum(categorical_feature_out_dims))
    set_cfg(cfg, 'model.static_categorical_autoencoder.decoder.output_dims', categorical_feature_out_dims)

    model = build_model(cfg.model.static_categorical_autoencoder,
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

    trainer = StaticCategoricalAETrainer(config=cfg,
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
    static_categorical_ae_trainer_main(cols=cols)
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from Trainer.train import Trainer
from DataLoaders.loaderbase import get_dataloaders
from Utils.train import save_ckpt, initialize_weights
from Utils.namespace import set_cfg
from Visualization import *
from Trainer.EHRSafe.utils import get_temporal_categorical_card


class TemporalCategoricalAETrainer(Trainer):
    def __init__(self, config, rank, **kwargs):
        super(TemporalCategoricalAETrainer, self).__init__(config=config, rank=rank, **kwargs)

        self.dataloaders = get_dataloaders(cfg=self.cfg,
                                           train_dataset=kwargs['train_dataset'],
                                           valid_dataset=kwargs['validation_dataset'],
                                           test_dataset=kwargs['test_dataset'],
                                           rank=self.rank,
                                           world_size=self.world_size,
                                           collate_fn=kwargs['collate_fn'])
        self.test_loss = {key: [] for key in self.loss_keys}
        self.logit_threshold = 0.5

        self.transformer = self.dataloaders.train.dataset.temporal_transformer
        self.tc_cols = self.dataloaders.train.dataset.tc_cols
        self.tl_cols = self.dataloaders.train.dataset.tl_cols

        temporal_categorical_card, categorical_feature_info = get_temporal_categorical_card(self.dataloaders.train.dataset,
                                                                                            self.transformer)
        self.categorical_feature_info = categorical_feature_info

        self.tc_binary_cols = [col.column_name for col in self.categorical_feature_info if col.column_type == 'Binary']

        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.diagnoses_prefix = self.cfg.preprocess.icd_code.diagnoses_prefix
        self.procedure_prefix = self.cfg.preprocess.icd_code.procedure_prefix
        self.proc_prefix = self.cfg.preprocess.proc_prefix

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

            tc_data, tn_data, tl_data, temporal_mask, condition = self._unpack_temporal_batch(batch)
            data = torch.cat([tc_data, tl_data], dim=-1)
            batch_size, seq_len, feature_dim = data.size()
            rep, x_hat = self.model(data)

            mask = torch.sum(data == self.cfg.dataloader.pad_value, dim=-1) == 0.

            loss = 0
            s_idx = 0
            for i, col in enumerate(self.tc_cols + self.tl_cols):
                _x_hat = x_hat[i]
                _x_hat = _x_hat.view(batch_size, seq_len, -1)
                dim = _x_hat.shape[-1]

                if col in self.tc_cols:
                    if col in self.tc_binary_cols:
                        _loss = self.bce_loss_fn(_x_hat.view(-1), data[:, :, s_idx].view(-1))
                    else:
                        _loss = self.ce_loss_fn(_x_hat.view(-1, dim),
                                                torch.argmax(data[:, :, s_idx: s_idx + dim], dim=-1).view(-1))
                    _loss = _loss * mask.view(-1)

                elif col in self.tl_cols:
                    _loss = self.bce_loss_fn(_x_hat.view(-1, dim), data[:, :, s_idx: s_idx + dim].contiguous().view(-1, dim))
                    _loss = _loss * mask.view(-1, 1)

                # if dim == 1:
                #     _loss = self.bce_loss_fn(_x_hat.view(-1), data[:, :, s_idx].view(-1))
                # else:
                #     _loss = self.ce_loss_fn(_x_hat.view(-1, dim), torch.argmax(data[:, :, s_idx: s_idx+dim], dim=-1).view(-1))
                # _loss = _loss * mask.view(-1)
                _loss = _loss.sum() / mask.sum()
                loss = loss + _loss
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
                tc_data, tn_data, tl_data, temporal_mask, condition = self._unpack_temporal_batch(batch)
                data = torch.cat([tc_data, tl_data], dim=-1)
                batch_size, seq_len, feature_dim = data.size()
                rep, x_hat = self.model(data)

                mask = torch.sum(data == self.cfg.dataloader.pad_value, dim=-1) == 0.

                loss = 0
                s_idx = 0
                for i, col in enumerate(self.tc_cols + self.tl_cols):
                    _x_hat = x_hat[i]
                    _x_hat = _x_hat.view(batch_size, seq_len, -1)
                    dim = _x_hat.shape[-1]

                    if col in self.tc_cols:
                        if col in self.tc_binary_cols:
                            _loss = self.bce_loss_fn(_x_hat.view(-1), data[:, :, s_idx].view(-1))
                        else:
                            _loss = self.ce_loss_fn(_x_hat.view(-1, dim),
                                                    torch.argmax(data[:, :, s_idx: s_idx + dim], dim=-1).view(-1))
                        _loss = _loss * mask.view(-1)

                    elif col in self.tl_cols:
                        _loss = self.bce_loss_fn(_x_hat.view(-1, dim),
                                                 data[:, :, s_idx: s_idx + dim].contiguous().view(-1, dim))
                        _loss = _loss * mask.view(-1, 1)

                    # if dim == 1:
                    #     _loss = self.bce_loss_fn(_x_hat.view(-1), data[:, :, s_idx].view(-1))
                    # else:
                    #     _loss = self.ce_loss_fn(_x_hat.view(-1, dim), torch.argmax(data[:, :, s_idx: s_idx+dim], dim=-1).view(-1))
                    # _loss = _loss * mask.view(-1)
                    _loss = _loss.sum() / mask.sum()
                    loss = loss + _loss
                    s_idx += dim

                total_losses['Total_Loss'] += loss.item()
                self._set_iterator_postfix(validation_iterator, total_losses['Total_Loss'])

            for key in self.loss_keys:
                self.validation_loss[key].append(total_losses[key] / len(self.dataloaders.valid))

    def eval_one_epoch(self, epoch):
        total_losses = {key: 0.0 for key in self.loss_keys}
        test_iterator = self._set_iterator(self.dataloaders.test, epoch, mode='test')

        transformer = self.dataloaders.test.dataset.temporal_transformer

        if not os.path.exists(self.cfg.path.plot_file_path):
            os.makedirs(self.cfg.path.plot_file_path, exist_ok=True)

        with torch.no_grad():
            target = []
            data_hat = []
            for batch in test_iterator:
                tc_data, tn_data, tl_data, temporal_mask, condition = self._unpack_temporal_batch(batch)
                data = torch.cat([tc_data, tl_data], dim=-1)
                batch_size, seq_len, feature_dim = data.size()
                rep, x_hat = self.model(data)

                mask = torch.sum(data == self.cfg.dataloader.pad_value, dim=-1) == 0.

                loss = 0
                s_idx = 0
                act_x_hat = []
                for i, col in enumerate(self.tc_cols+self.tl_cols):
                    _x_hat = x_hat[i]
                    _x_hat = _x_hat.view(batch_size, seq_len, -1)
                    dim = _x_hat.shape[-1]

                    if col in self.tc_cols:
                        if col in self.tc_binary_cols:
                            _loss = self.bce_loss_fn(_x_hat.view(-1), data[:, :, s_idx].view(-1))
                            _act_x_hat = torch.sigmoid(_x_hat)
                            _act_x_hat = (_act_x_hat >= self.logit_threshold).float()
                            act_x_hat.append(_act_x_hat)

                        else:
                            _loss = self.ce_loss_fn(_x_hat.view(-1, dim),
                                                    torch.argmax(data[:, :, s_idx: s_idx + dim], dim=-1).view(-1))
                            act_x_hat.append(torch.softmax(_x_hat, dim=-1))

                        _loss = _loss * mask.view(-1)
                    elif col in self.tl_cols:
                        _loss = self.bce_loss_fn(_x_hat.view(-1, dim), data[:, :, s_idx: s_idx + dim].contiguous().view(-1, dim))
                        _loss = _loss * mask.view(-1, 1)

                        _act_x_hat = torch.sigmoid(_x_hat)
                        _act_x_hat = (_act_x_hat >= self.logit_threshold).float()
                        act_x_hat.append(_act_x_hat)

                    _loss = _loss.sum() / mask.sum()
                    loss = loss + _loss

                    s_idx += dim

                total_losses['Total_Loss'] += loss.item()
                self._set_iterator_postfix(test_iterator, total_losses['Total_Loss'])

                act_x_hat = torch.concatenate(act_x_hat, dim=-1).view(-1, seq_len, feature_dim)
                x_hat = torch.concatenate(x_hat, dim=-1).view(-1, seq_len, feature_dim)
                target.append(data)
                data_hat.append(act_x_hat)
            target = torch.concatenate(target, dim=0)
            data_hat = torch.concatenate(data_hat, dim=0)

            mask = target != self.cfg.dataloader.pad_value
            target = transformer.inverse_transform(target.view(-1, feature_dim).detach().cpu().numpy(), self.categorical_feature_info)
            data_hat = transformer.inverse_transform(data_hat.view(-1, feature_dim).detach().cpu().numpy(), self.categorical_feature_info)

            mask = mask.view(-1, feature_dim)
            mask = mask[:, 0].view(-1, 1).repeat(1, target.shape[-1])
            mask = mask.detach().cpu().numpy()
            target = pd.DataFrame(np.where(mask, target, np.nan), columns=target.columns)
            data_hat = pd.DataFrame(np.where(mask, data_hat, np.nan), columns=data_hat.columns)

            for col in target.columns:
                _col = col.replace('/', ' ').replace('>', ' ')
                countplot_categorical_feature(data1=target,
                                              data2=data_hat,
                                              col=col,
                                              stat='proportion',
                                              label1='Real',
                                              label2='Reconstructed',
                                              title=_col,
                                              save_path=self.cfg.path.plot_file_path + f'countplot_{_col}_epoch_{epoch}.png')

            for key in self.loss_keys:
                self.test_loss[key].append(total_losses[key] / len(self.dataloaders.test))

    def _unpack_temporal_batch(self, batch):
        tc_data = batch[1].to(self.device)
        tn_data = batch[3].to(self.device)
        tl_data = batch[5].to(self.device)
        temporal_mask = batch[7].to(self.device)
        condition = None
        return tc_data, tn_data, tl_data, temporal_mask, condition


def temporal_categorical_ae_trainer_main(cols: List[str]=None):
    import config_manager

    from Models.EHRSafe.TemporalCategoricalAutoEncoder import build_model
    from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset

    from Utils.reproducibility import lock_seed
    from Utils.train import set_loss_fn, set_scheduler, set_optimizer

    config_manager.load_config()
    cfg = config_manager.config
    cfg.log.wandb.name = f'{cfg.log.time}_{cfg.log.ipaddr}'
    cfg.log.wandb.tags = ['TemporalCategoricalAE']

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

    # update model config following the dataset
    categorical_feature_out_dims, categorical_feature_info = get_temporal_categorical_card(train_dataset,
                                                                                           train_dataset.temporal_transformer)

    set_cfg(cfg, 'model.temporal_categorical_autoencoder.encoder.input_dim', sum(categorical_feature_out_dims))
    set_cfg(cfg, 'model.temporal_categorical_autoencoder.decoder.output_dims', categorical_feature_out_dims)

    model = build_model(cfg.model.temporal_categorical_autoencoder,
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

    trainer = TemporalCategoricalAETrainer(config=cfg,
                                           rank=cfg.device_num,
                                           world_size=world_size,
                                           model=model,
                                           collate_fn=None,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           scheduler=scheduler,
                                           train_dataset=train_dataset,
                                           validation_dataset=validation_dataset,
                                           test_dataset=test_dataset)
    trainer.run_epochs()


if __name__ == '__main__':
    cols = None
    temporal_categorical_ae_trainer_main(cols=cols)
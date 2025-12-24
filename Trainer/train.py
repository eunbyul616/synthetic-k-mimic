from tqdm import tqdm
import torch
from Trainer.trainbase import TrainBase

from Utils.train import EarlyStopping, load_ckpt, save_ckpt
from DataLoaders.loaderbase import get_dataloaders


class Trainer(TrainBase):
    def __init__(
            self,
            config,
            rank: int = 0,
            **kwargs
    ):
        super(Trainer, self).__init__(config, **kwargs)

        self.cfg = config

        self.rank = rank
        self.device = torch.device(f'cuda:{self.rank}')
        self.world_size = kwargs['world_size']

        self.dataloaders = get_dataloaders(cfg=self.cfg,
                                           train_dataset=kwargs['train_dataset'],
                                           valid_dataset=kwargs['validation_dataset'],
                                           rank=self.rank,
                                           world_size=self.world_size,
                                           collate_fn=kwargs['collate_fn'])

        self.model_name = self.cfg.train.model_name
        self.model_save_path = self.cfg.path.ckpt_path
        self.save_name = self.cfg.log.time

        self.model = kwargs['model']
        self.loss_fn = kwargs['loss_fn']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.scheduler_flag = False

        # Load the model from the checkpoint if it exists, otherwise start from scratch
        self.start_epoch = load_ckpt(
            self.cfg,
            self.model,
            self.rank,
            optimizer=self.optimizer,
            scheduler=self.scheduler)

        self.total_epochs = self.cfg.train.general.num_epochs
        # Early Stopping Watcher
        self.train_watcher = EarlyStopping(
            patience=self.cfg.train.general.early_stopping.patience,
            verbose=self.cfg.train.general.early_stopping.verbose
        )

        self.loss_keys = self.cfg.train.general.keys
        self.train_loss = {key: [] for key in self.loss_keys}
        self.validation_loss = {key: [] for key in self.loss_keys}

        self.target_key = self.cfg.train.general.early_stopping.target_key

    def run_epochs(self):
        for epoch in range(self.start_epoch, self.total_epochs):
            self.model.train()
            self.train_one_epoch(epoch)
            self.model.eval()
            self.validate_one_epoch(epoch)
            if self.scheduler is not None:
                stop_flag, self.scheduler_flag = self.train_watcher.check(epoch,
                                                                          self.validation_loss[self.target_key][-1])
                if stop_flag:
                    break

                # if self.scheduler_flag:
                #     self.scheduler.step()
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
                rank=self.rank
            )

    def _set_iterator(self, loader, epoch, mode='Train'):
        if (self.rank == 0) or (self.rank == self.cfg.device_num):
            return tqdm(loader, desc=f'{mode} Epoch: {epoch}', dynamic_ncols=True)
        else:
            return tqdm(loader, disable=True)

    def _set_iterator_postfix(self, iterator, loss):
        iterator.set_postfix(loss=loss / (iterator.n + 1),
                             lr=self.optimizer.param_groups[0]['lr'])

    def train_one_epoch(self, epoch):
        train_iterator = self._set_iterator(self.dataloaders.train, epoch, mode='Train')

        train_cost = 0

        for train_x, train_y in train_iterator:
            assert torch.isnan(train_x).sum() == 0, "Data Contains NaN in train_x"
            self.optimizer.zero_grad()

            train_x = train_x.to(torch.device(f'cuda:{self.rank}'), non_blocking=True)
            train_y = train_y.to(torch.device(f'cuda:{self.rank}'), non_blocking=True)

            out = self.model(train_x)

            loss = self.loss_fn(out, train_y)
            train_cost += loss.item()

            loss.backward()

            if self.cfg['pretrain'].train.general.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.paramter(), max_norm=1.0)

            self.optimizer.step()

            train_iterator.set_postfix(loss=train_cost / (train_iterator.n + 1),
                                       lr=self.optimizer.param_groups[0]['lr'])

        self.train_loss['Total_Loss'].append(train_cost / len(self.dataloaders.train))

    def validate_one_epoch(self, epoch):
        validation_iterator = self._set_iterator(self.dataloaders.valid, epoch, mode='Valid')

        valid_cost = 0

        with torch.no_grad():
            for valid_x, valid_y in validation_iterator:
                assert torch.isnan(valid_x).sum() == 0, "Data Contains NaN in valid_x"

                valid_x = valid_x.to(torch.device(f'cuda:{self.rank}'), non_blocking=True)
                valid_y = valid_y.to(torch.device(f'cuda:{self.rank}'), non_blocking=True)

                out = self.model(valid_x)

                loss = self.loss_fn(out, valid_y)
                valid_cost += loss.item()

                self._set_iterator_postfix(validation_iterator, valid_cost)

            self.validation_loss['Total_Loss'].append(valid_cost / len(self.dataloaders.valid))

    def eval_one_epoch(self, epoch):
        pass


import os
import yaml
import wandb
from types import SimpleNamespace


class WandbLogger:
    def __init__(self,
                 cfg: SimpleNamespace,
                 rank: int = 0,
                 start_epoch: int = 0,
                 ):
        self.cfg = cfg
        self.rank = rank
        self.start_epoch = start_epoch
        self.log_flag = ((rank == 0 and cfg.log.flag and cfg.log.wandb_flag) or
                         (rank == cfg.device_num and cfg.log.flag and cfg.log.wandb_flag))

    def _init(self):
        log_path = self.cfg.path.wandb_log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        config_saved_path = self.cfg.path.base_config_file_path.replace('save_name', self.cfg.log.time)

        artifact = wandb.Artifact(name='total_configuration', type='config')
        artifact.add_file(config_saved_path)
        with open(config_saved_path, 'r') as f:
            config_data = yaml.safe_load(f)

        wandb.init(project=self.cfg.log.wandb.project,
                   entity=self.cfg.log.wandb.entity,
                   name=self.cfg.log.wandb.name,
                   tags=self.cfg.log.wandb.tags,
                   dir=log_path,
                   config=config_data)


    def log(self, train_loss: dict, validation_loss: dict, current_epoch: int, **kwargs):
        """
        Log the training and validation loss to wandb.
        Initialize the wandb session if it is the first epoch.
        Args:
            train_loss (dict): training loss
            validation_loss (dict): validation loss
            current_epoch (int): current epoch
        Returns:
            None
        """
        if self.log_flag:
            if current_epoch == self.start_epoch:
                self._init()

            log_dict = {}
            for key in train_loss.keys():
                log_dict[f"Train {key}"] = train_loss[key][-1]
                log_dict[f"Validation {key}"] = validation_loss[key][-1]

            if 'model' in kwargs.keys():
                model = kwargs['model']
                wandb.watch(model, log='all')

            if 'metrics' in kwargs.keys():
                metrics = kwargs['metrics']
                for key in metrics.keys():
                    log_dict[f"Metrics {key}"] = metrics[key][-1]

            wandb.log(log_dict, step=current_epoch)

    def cleanup(self):
        """
        Finish the wandb session.
        """
        if self.log_flag:
            wandb.finish()
            print("wandb session finished.")


import os
import shutil
import logging
from pathlib import Path

import numpy as np
from copy import deepcopy
from tqdm import tqdm
from types import SimpleNamespace

from Utils.namespace import compare_namespaces_recursive, update_cfg, _load_yaml
from Utils.file import get_all_files_recursive
import Loss.generation as generation_loss

import torch
import torch.nn as nn
from torch.optim import Optimizer


class Lookahead(Optimizer):
    def __init__(
            self,
            optimizer,
            k=5,
            alpha=0.5
    ):
        """
        Lookahead optimizer
        Args:
            optimizer: base optimizer
            k (int): number of lookahead steps
            alpha (float): slow weights update rate

        Usage:
            base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer = Lookahead(optimizer=base_optimizer, k=5, alpha=0.5)

            optimizer.zero_grad()

            ...

            optimizer.step()
        """
        super(Lookahead, self).__init__(optimizer.param_groups, optimizer.defaults)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")

        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups

        self.state = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p] = {'slow_param': p.data.clone(), 'step': 0}

    def step(self, **kwargs):
        """
        """

        loss = self.optimizer.step()

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['step'] += 1

                    if state['step'] % self.k == 0:
                        slow_param = state['slow_param']
                        slow_param.add_(p.data - slow_param, alpha=self.alpha)
                        p.data.copy_(slow_param)

        return loss

    def zero_grad(self, **kwargs):
        """
        Clears the gradients of all optimized parameters.
        Args:
            kwargs: additional arguments to pass to the base optimizer's zero_grad method
        """
        self.optimizer.zero_grad()


def set_loss_fn(config):
    """
    Set loss function for training
    Args:
        config: configuration for loss function
    """
    name = config.name
    if name == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif name == 'bce':
        loss_fn = nn.BCELoss()
    elif name == 'mae':
        loss_fn = nn.L1Loss()
    elif name == 'mse':
        loss_fn = nn.MSELoss()
    elif name == 'generator':
        loss_fn = generation_loss.GeneratorLoss()
    elif name == 'discriminator':
        loss_fn = generation_loss.DiscriminatorLoss()

    elif name == 'wgan_generator':
        loss_fn = generation_loss.WGANGeneratorLoss()
    elif name == 'wgan_discriminator':
        loss_fn = generation_loss.WGANDiscriminatorLoss()

    else:
        raise ValueError("Invalid loss function")

    return loss_fn


def set_optimizer(
        model_params,
        config,
        apply_lookahead=True
):
    """
    Set optimizer for training.
    # TODO: Add more optimizers for flexibility e.g. RMSprop, Adagrad, SAM etc.
    Args:
        model_params: model parameters for update
        config: configuration for optimizer
        apply_lookahead: whether to use lookahead for optimizer

    Returns:
        optimizer

    """
    name = config.name
    lr = float(config.lr)
    # weight_decay = float(config.weight_decay)
    if name == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=lr)
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_params),
                                      lr=lr,
                                      weight_decay=float(config.weight_decay),
                                      betas=(config.beta1, config.beta2))
    elif name == 'sgd':
        optimizer = torch.optim.SGD(model_params, lr=lr, momentum=float(config.momentum))
    else:
        raise ValueError("Invalid optimizer")
    if apply_lookahead:
        print('Applying Lookahead for optimizer')
        optimizer = Lookahead(optimizer,
                              k=config.lookahead.k,
                              alpha=config.lookahead.alpha)
    return optimizer


def set_scheduler(optimizer, config):
    """
    Set scheduler for optimizer.
    # TODO: Add more scheduler for flexibility

    Args:
        optimizer:
        config: configuration for scheduler

    Returns:
        scheduler

    """
    name = config.name
    if name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif name == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    elif name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
    elif name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    else:
        raise ValueError("Invalid scheduler")
    return scheduler


class EarlyStopping:
    """
    Early stopping logic for training when the validation loss does not improve after certain epochs.
    """

    def __init__(
            self,
            patience=10,
            verbose=True
    ):
        """
        Args:
            patience: number of epochs to wait before stopping the training
            verbose: whether to print the early stopping message
        """
        self.patience = patience
        self.verbose = verbose

        self.stop_flag = False
        self.scheduler_flag = False
        self.best_loss = float('inf')
        self.prior_loss = float('inf')
        self.wait = 0

    def check(self, epoch, loss) -> tuple:
        """
        Check if the loss has improved or not
            If the loss has not improved for half of the patience, the scheduler will be activated for further training.

        Args:
            epoch: current epoch of training step
            loss: current(last) validation loss

        Returns:
            bool: whether the model has improved or not

        """
        if loss < self.prior_loss:
            self.wait = 0
        else:
            self.wait += 1
        self.prior_loss = loss

        # if loss < self.best_loss:
        #     self.best_loss = loss
        #     self.wait = 0

        self.stop_flag = self.wait >= self.patience
        self.scheduler_flag = True if self.wait >= self.patience // 2 else False

        if self.stop_flag:
            if self.verbose:
                print(f"Early stopping at epoch {epoch}")
        else:
            if self.verbose:
                print(f"watcher.wait: {self.wait} / watcher.patience: {self.patience}")

        return self.stop_flag, self.scheduler_flag


def initialize_weights(m) -> None:
    """
    Initialize weights for model.

    Args:
        m: Model

    Returns:
        Weights initialized model

    Usage:
        model = DummyModel()
        model.apply(initialize_weights)

    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight.data, -0.1, 0.1)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0)
    else:
        pass


def save_ckpt(
        cfg: SimpleNamespace,
        epoch: int,
        validation_loss: list,
        states: dict,
        save_condition: str = 'last',
        start_epoch: int = 0,
        rank: int = 0) -> None:
    """
    Save model checkpoint
    Args:
        cfg : configuration namespace
        epoch: current epoch
        validation_loss: list of validation loss (whichever metric for tracking the model performance)
        states:
        rank: rank of gpu

    Returns:
        save_flag: None

    """

    def _select_save_condition() -> bool:
        if save_condition == 'lower':
            return validation_loss[-1] == min(validation_loss)
        elif save_condition == 'higher':
            return validation_loss[-1] == max(validation_loss)
        elif save_condition == 'last':
            return validation_loss[-2] > validation_loss[-1]

    if (rank == 0) or (rank == cfg.device_num):
        target_path = cfg.path.ckpt_config_file_path

        # if scheduler is none remove scheduler from states
        states.pop('scheduler') if states['scheduler'] is None else None

        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)

        # copy config.yaml from hydra output directory to model_save_path
        shutil.copyfile(cfg.path.base_config_file_path,
                        os.path.join(target_path, 'config.yaml'))

        # handle DataParallel model save case
        if hasattr(states['model'], 'module'):
            states['state_dict'] = states['model'].module.state_dict()
        else:
            states['state_dict'] = states['model'].state_dict()

        # move model state_dict to cpu before saving
        states['state_dict'] = {k: v.cpu() for k, v in states['state_dict'].items()}

        # save current state of model
        print(f"Saving current model checkpoint")
        current_ckpt_path = os.path.join(target_path, 'checkpoint.pth.tar')
        best_ckpt_path = os.path.join(target_path, 'checkpoint_best.pth.tar')

        states.pop('model', None)
        if states.get('scheduler') is None:
            states.pop('scheduler', None)

        torch.save(states, current_ckpt_path)

        if epoch > 0:
            if _select_save_condition():
                if epoch > 0 and epoch > start_epoch:
                    if save_condition == 'lower':
                        print(f"*** Saving best model at {epoch}: {min(validation_loss[:-1])} => {validation_loss[-1]} ***")
                    elif save_condition == 'higher':
                        print(f"*** Saving best model at {epoch}: {max(validation_loss[:-1])} => {validation_loss[-1]} ***")
                    elif save_condition == 'last':
                        print(f"*** Saving best model at {epoch}: {validation_loss[-2]} => {validation_loss[-1]} ***")
                else:
                    if save_condition == 'lower':
                        print(f"*** Saving best model at {epoch}: {validation_loss[-1]} ***")
                    elif save_condition == 'higher':
                        print(f"*** Saving best model at {epoch}: {validation_loss[-1]} ***")
                    elif save_condition == 'last':
                        print(f"*** Saving best model at {epoch}: {validation_loss[-1]} ***")

                shutil.copyfile(current_ckpt_path, best_ckpt_path)
                update_cfg(cfg, ['train.loss.best_value'], [validation_loss[-1]], save=True)


def load_ckpt(cfg,
              model: nn.Module,
              rank: int = 0,
              **kwargs
              ) -> int:
    """
    Load model checkpoint with same configuration

    Args:
        cfg: configuration
        model:
        rank:

    **kwargs:
        optimizers and schedulers


    Returns:
        start_epoch: start epoch for training

    """
    try:
        load_condition = getattr(cfg.metrics, cfg.train.loss.name).is_better
    except AttributeError:
        print(f"Cannot find '{cfg.train.loss.name}' in metrics\n Using 'latest' condition for loading checkpoint")
        load_condition = 'latest'

    start_epoch = 0

    def _select_configuration(
            configuration_list: list,
            metrics: list
    ) -> str or None:
        if len(configuration_list) == 0:
            return None

        if load_condition == 'latest':
            config_path = sorted(configuration_list)[-1]
        elif load_condition == 'lower':
            idx = np.argmin(metrics)
            config_path = configuration_list[idx]
        elif load_condition == 'higher':
            idx = np.argmax(metrics)
            config_path = configuration_list[idx]
        else:
            raise ValueError

        return config_path

    def _retrieve_same_configuration(
            model_saved_path,
            current_model_config_path,
            unnecessary_keys: list[str] = None
    ) -> str or None:
        """
        Retrieve the latest checkpoint with same configuration
        Args:
            model_saved_path: root path for saved checkpoints (/results/checkpoints)
            current_model_config_path: current model configuration path (/log/hydra/{save_name}/.hydra/config.yaml)
            unnecessary_keys: unnecessary keys to remove from configuration for comparison

        Returns:
            latest_config: latest checkpoint path with same configuration if found, otherwise None
        """

        def _compare_configuration(trained_config_path, keys: list[str] = None) -> tuple[bool, float | None]:
            def _remove_unnecessary_keys(config):
                copied_config = update_cfg(deepcopy(config), keys, [None] * len(keys), save=False)
                return copied_config

            comp_config = _load_yaml(trained_config_path)
            base_config = _load_yaml(current_model_config_path)

            # remove unnecessary keys from configurations
            dict_diff = compare_namespaces_recursive(_remove_unnecessary_keys(comp_config),
                                                     _remove_unnecessary_keys(base_config))

            if dict_diff == {}:
                try:
                    value = comp_config.train.loss.best_value
                except AttributeError:
                    value = -1.0
                return True, value
            else:
                return False, None

        trained_config_list = get_all_files_recursive(model_saved_path, ext='.yaml')

        if len(trained_config_list) == 0:
            return None
        same_configuration_list = []
        metric_list = []

        for trained_config in tqdm(trained_config_list, desc='Comparing configurations'):
            flag, metric = _compare_configuration(trained_config, unnecessary_keys)
            if flag:
                same_configuration_list.append(trained_config)
                metric_list.append(metric)

        return _select_configuration(same_configuration_list, metric_list)

    checkpoint_saved_root = cfg.path.ckpt_path

    # configuration path for current setting
    current_config_path = cfg.path.base_config_file_path
    # configuration path for the latest checkpoint with same configuration

    if hasattr(cfg, 'ignore'):
        src_config_path = _retrieve_same_configuration(checkpoint_saved_root,
                                                       current_config_path,
                                                       unnecessary_keys=cfg.ignore.keys)
    else:
        src_config_path = _retrieve_same_configuration(checkpoint_saved_root,
                                                       current_config_path)

    if src_config_path is None:
        print(f"No checkpoint found based on '{current_config_path}'... Starting from scratch")
        return start_epoch

    src_checkpoint_path = os.path.join(checkpoint_saved_root,
                                       '/'.join(Path(src_config_path).parts[-3:-1]))
    dst_checkpoint_path = os.path.join(checkpoint_saved_root,
                                       cfg.log.time)

    if os.path.isdir(src_checkpoint_path):
        print(f"Loading checkpoint at rank: {rank}")
        # copy the latest checkpoint with same configuration to the new directory
        if not os.path.exists(dst_checkpoint_path):
            shutil.copytree(src=src_checkpoint_path, dst=dst_checkpoint_path, dirs_exist_ok=True)

        if os.path.isfile(os.path.join(dst_checkpoint_path, 'checkpoint_best.pth.tar')):
            checkpoint = torch.load(os.path.join(dst_checkpoint_path, 'checkpoint_best.pth.tar'),
                                    map_location=f'cuda:{rank}', weights_only=False)
        else:
            print('No best checkpoint found, loading the latest checkpoint')
            return start_epoch

        start_epoch = checkpoint['epoch']
        # if model is wrapped with DDP, add 'module.' to the checkpoint's state_dict keys (
        if hasattr(model, 'module'):
            model.load_state_dict({'module.' + k: v for k, v in checkpoint['state_dict'].items()})
        else:
            model.load_state_dict(checkpoint['state_dict'])

        # load optimizers and schedulers
        for key, obj in kwargs.items():
            if 'optimizer' in key:
                obj.load_state_dict(checkpoint[f'{key}'])
            if ('scheduler' in key) and (obj is not None):
                obj.load_state_dict(checkpoint[f'{key}'])

        # if rank == 0:
        print(f"Loaded checkpoint '{src_checkpoint_path}' (epoch {start_epoch})")
    else:
        # if rank == 0:
        print(f"No checkpoint found at '{src_checkpoint_path}'")

    return start_epoch


def setup_logger(
        rank,
        log_dir='logs',
        log_level=logging.DEBUG):
    """
    Setup logger for training process
    Args:
        rank:
        log_dir:
        log_level:
    Returns:
        logger for debugging
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(log_level)
    fh = logging.FileHandler(f"{log_dir}/rank{rank}.log")
    fh.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


@torch.no_grad()
def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm
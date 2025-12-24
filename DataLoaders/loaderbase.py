from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, DistributedSampler

from typing import Optional
from Utils.reproducibility import dataloader_seed_worker


@dataclass
class SamplerCollection:
    train: Optional[DistributedSampler] = None
    valid: Optional[DistributedSampler] = None
    test: Optional[DistributedSampler] = None


@dataclass
class DataLoaderCollection:
    train: DataLoader
    valid: DataLoader
    test: Optional[DataLoader] = None


def get_dataloaders(
        cfg,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        rank: int = 0,
        world_size: int = 1,
        collate_fn: callable = 'none',
        **kwargs
):
    """

    Args:
        cfg:
        train_dataset: training dataset
        valid_dataset: validation dataset
        test_dataset: test dataset (optional)
        rank (int): rank of the process
        world_size(int): total number of processes
        collate_fn (callable, optional): collate function to apply to the dataloader

    Returns:
        dataloaders
    """
    if cfg.train.general.distributed:
        train_sampler = DistributedSampler(dataset=train_dataset,
                                           num_replicas=world_size,
                                           rank=rank,
                                           shuffle=cfg.dataloader.train_shuffle)
        valid_sampler = DistributedSampler(dataset=valid_dataset,
                                           num_replicas=world_size,
                                           rank=rank,
                                           shuffle=cfg.dataloader.validation_shuffle)

        g = torch.Generator()
        g.manual_seed(cfg.train.general.seed + rank)

    else:
        if 'train_sampler' in kwargs:
            train_sampler = kwargs['train_sampler']
        else:
            train_sampler = None
        if 'valid_sampler' in kwargs:
            valid_sampler = kwargs['valid_sampler']
        else:
            valid_sampler = None

        g = None

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.dataloader.batch_size,
                              pin_memory=True,
                              sampler=train_sampler,
                              worker_init_fn=dataloader_seed_worker(rank),
                              generator=g,
                              collate_fn=collate_fn,
                              drop_last=cfg.dataloader.drop_last
                              )

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=cfg.dataloader.batch_size,
                              pin_memory=True,
                              sampler=valid_sampler,
                              worker_init_fn=dataloader_seed_worker(rank),
                              generator=g,
                              collate_fn=collate_fn,
                              drop_last=cfg.dataloader.drop_last
                              )

    if test_dataset is not None:
        if cfg.train.general.distributed:
            test_sampler = DistributedSampler(dataset=test_dataset,
                                              num_replicas=world_size,
                                              rank=rank,
                                              shuffle=cfg.dataloader.test_shuffle)
        else:
            if 'test_sampler' in kwargs:
                test_sampler = kwargs['test_sampler']
            else:
                test_sampler = None

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=cfg.dataloader.batch_size,
                                 pin_memory=True,
                                 sampler=test_sampler,
                                 worker_init_fn=dataloader_seed_worker(rank),
                                 generator=g,
                                 collate_fn=collate_fn,
                                 drop_last=cfg.dataloader.drop_last
                                 )
    else:
        test_loader = None

    # samplers = SamplerCollection(train_sampler, valid_sampler, test_sampler)
    dataloaders = DataLoaderCollection(train_loader, valid_loader, test_loader)

    # return samplers, dataloaders
    return dataloaders

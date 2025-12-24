import os
import random
import numpy as np

import torch
from torch.backends import cudnn


def lock_seed(seed: int = 42,
              multi_gpu=True,
              activate_cudnn=False
              ) -> None:
    """
    Lock seed for reproducibility and reduce training time.

        Considerations:
        Seed:
            - numpy.random.seed()
            - torch.manual_seed()
            - torch.cuda.manual_seed()
            - torch.cuda.manual_seed_all() if multi-GPU
            - torch.backends.cudnn.deterministic = True
            - torch.backends.cudnn.benchmark = False
        Training speed:
            - cuda.allow_tf32 = True
            - cudnn.allow_tf32 = True

    See https://pytorch.org/docs/stable/notes/randomness.html for more information.

    Args:
        seed: seed for reproducibility
        multi_gpu: whether to use multi-GPU for training
        activate_cudnn: whether to activate cudnn for training

    Returns:
        None

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) if multi_gpu else None
        torch.cuda.enabled = True
        torch.cuda.allow_tf32 = True  # allowing TensorFloat32 for faster training
        if activate_cudnn:
            cudnn.enabled = True
            cudnn.allow_tf32 = True  # allowing TensorFloat32 for faster training
            cudnn.benchmark = False  # turn on for faster training ( if turned on, may be not reproducible )
            cudnn.deterministic = True  # turn on for reproducibility ( if turned on, slow down training )
        else:
            cudnn.enabled = False
        torch.set_float32_matmul_precision("high")

    else:
        print("cuda is not available")


def guarantee_reproducibility(seed: int = 42,
                              multi_gpu: bool = True,
                              device_number: str = "None",
                              activate_cudnn: bool = False
                              ) -> torch.device:
    """
    Lock seed for reproducibility and reduce training time.

    Args:
        seed: seed for reproducibility
        multi_gpu: whether to use multi-GPU for training
        device_number: device number for training
        activate_cudnn: whether to activate cudnn for training
    Returns:
        torch.device
    """
    # GPU Setting
    total_gpu_memory = 0

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if device_number == "None":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_number
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("----- GPU INFO -----\nDevice:", DEVICE)
    print("Count of using GPUs:", torch.cuda.device_count())
    print("Current cuda device:", torch.cuda.current_device())

    gpu_ids = list(map(str, list(range(torch.cuda.device_count()))))
    for gpu_id in gpu_ids:
        total_gpu_memory += torch.cuda.get_device_properties(
            "cuda:" + gpu_id
        ).total_memory
    print("Total GPU Memory :", total_gpu_memory, "\n--------------------")

    # Lock seed for reproducibility
    lock_seed(seed, multi_gpu, activate_cudnn)

    return DEVICE


def dataloader_seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

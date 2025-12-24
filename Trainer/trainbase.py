from abc import *


class TrainBase(metaclass=ABCMeta):
    def __init__(self, config, **kwargs):
        self.cfg = config

        self.start_epoch = 0

    @abstractmethod
    def run_epochs(self):
        """
        Run epochs for Training & Validation
            wandb logging, model saving ,logging if needed
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def train_one_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def validate_one_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def eval_one_epoch(self, epoch):
        raise NotImplementedError

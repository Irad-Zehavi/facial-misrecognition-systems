import warnings
from abc import ABC
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tqdm import tqdm

from src.data.core.smart_dataset import SmartDataset, BATCH_SIZE

warnings.filterwarnings('ignore', '.*Consider increasing the value of the `num_workers` argument.*')  # Haven't managed to improve accuracy yet


class DataModule(LightningDataModule, ABC):
    def __init__(self, batch_size=BATCH_SIZE):
        super(DataModule, self).__init__()
        self.batch_size = batch_size
        self.fit_dataset: Optional[SmartDataset] = None
        self.train_dataset: Optional[SmartDataset] = None
        self.val_dataset: Optional[SmartDataset] = None
        self.test_dataset: Optional[SmartDataset] = None
        self.save_hyperparameters('batch_size')

    @classmethod
    def load(cls, stage=None, *args, **kwargs):
        data_module = cls(*args, **kwargs)
        data_module.setup(stage)
        return data_module

    @property
    def labels(self):
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in {'fit', 'validate', None}:
            if not self.fit_dataset:
                self.fit_dataset = self._setup_fit_dataset()
            if not self.train_dataset or not self.val_dataset:
                self.train_dataset, self.val_dataset = self.fit_dataset.fraction_random_split(0.1)
        if stage in {'test', None} and not self.test_dataset:
            self.test_dataset = self._setup_test_dataset()

    def _setup_fit_dataset(self) -> SmartDataset:
        raise NotImplementedError

    def _setup_test_dataset(self) -> SmartDataset:
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_dataset.dataloader(shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataset.dataloader(batch_size=self.batch_size)

    def fit_dataloader(self, *args, **kwargs):
        return self.fit_dataset.dataloader(batch_size=self.batch_size, *args, **kwargs)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataset.dataloader(batch_size=self.batch_size)

    def __str__(self):
        return type(self).__name__

    @property
    def sample_shape(self):
        raise NotImplementedError  # not all subclasses have to implement

    def compute_normalization_params_batched(self):
        sum_, sum_of_squared, n = 0, 0, 0
        for X, y in tqdm(self.train_dataloader(), desc='Computing mean and StD'):
            num_dims = len(X.shape)
            reduced_dims = (0, num_dims - 2, num_dims - 1)  # Reduce batch, width and height dimensions
            sum_ += X.sum(dim=reduced_dims)
            sum_of_squared += X.pow(2).sum(dim=reduced_dims)
            n += X.shape[0] * X.shape[-2] * X.shape[-1]

        mean = sum_ / n
        std = ((sum_of_squared / n) - (mean ** 2)).sqrt()
        return mean, std

    def compute_normalization_params(self):
        imgs = torch.stack([img for img, label in self.train_dataloader()])
        num_dims = len(imgs.shape)
        reduced_dims = (0, num_dims - 2, num_dims - 1)  # Reduce batch, width and height dimensions
        return imgs.mean(dim=reduced_dims), imgs.std(dim=reduced_dims)  # reducing all dims except for color

    def print_stats(self):
        self.setup()
        sample = self.train_dataset[0]
        X, y = sample[:-1], sample[-1]
        print("{} training samples, {} test samples".format(len(self.train_dataset), len(self.test_dataset)))
        print("Shape of X: ", [x.shape for x in X])
        print("Pixel range: min: {}, max: {}".format(X[0].min(), X[0].max()))
        print("Channel distributions: mean: {}, std: {}".format(*self.compute_normalization_params_batched()))
        if self.labels:
            print("Labels ({}): {}".format(len(self.labels), self.labels))

    @property
    def normalizer(self):
        """
        Override to support normalization
        """
        return None

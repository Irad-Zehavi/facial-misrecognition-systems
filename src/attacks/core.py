from abc import ABC, abstractmethod
from typing import List, Union

from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from src.data.core.data_module import DataModule
from src.data.core.smart_dataset import SmartDataset


class Attack(HyperparametersMixin, ABC, object):
    def __init__(self, logger=None, *args, **kwargs):
        super(Attack, self).__init__(*args, **kwargs)
        self.logger = logger

    @abstractmethod
    def attack_success_test_dataset(self, clean_test_dataset) -> Union[SmartDataset, List[SmartDataset]]:
        pass


class AttackData(DataModule):
    def __init__(self, clean_data: DataModule, attack: Attack, *args, **kwargs):
        super(AttackData, self).__init__(batch_size=clean_data.batch_size, *args, **kwargs)
        self.clean_data = clean_data
        self.attack = attack

    def setup(self, stage=None):
        self.clean_data.setup(stage)
        super(AttackData, self).setup(stage)

        if not self.attack.logger and self.trainer:
            self.attack.logger = self.trainer.logger
        if stage in {'fit', 'validate', 'test', None}:
            self.attack_success_test_dataset = self.attack.attack_success_test_dataset(self.clean_data.test_dataset)

    def _setup_fit_dataset(self) -> SmartDataset:
        return self.clean_data.fit_dataset

    def _setup_test_dataset(self) -> SmartDataset:
        return self.clean_data.test_dataset

    def attack_success_test_dataloader(self):
        if isinstance(self.attack_success_test_dataset, list):
            return [ds.dataloader(batch_size=self.batch_size) for ds in self.attack_success_test_dataset]
        return self.attack_success_test_dataset.dataloader(batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        clean_tests = self.clean_data.test_dataloader()
        asr_tests = self.attack_success_test_dataloader()
        if not isinstance(clean_tests, list):
            clean_tests = [clean_tests]
        if not isinstance(asr_tests, list):
            asr_tests = [asr_tests]
        return clean_tests + asr_tests

    def sample_shape(self):
        return self.clean_data.sample_shape

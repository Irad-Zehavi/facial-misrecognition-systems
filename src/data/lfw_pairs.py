from abc import ABC, abstractmethod

from torchvision import datasets
from torchvision.datasets.lfw import _LFW
from torchvision.transforms import RandomHorizontalFlip, ToTensor, Compose

from src.core.utils import DATA_PATH
from src.data.core.data_module import DataModule
from src.data.core.smart_dataset import SmartDataset


class LFWCroppedMixin(_LFW, ABC):
    def __init__(self, root, image_set='cropped', *args, **kwargs):
        self.file_dict['cropped'] = ('lfw_cropped', *self.file_dict['original'][1:])
        super(LFWCroppedMixin, self).__init__(root, image_set=image_set, *args, **kwargs)


class LFWPairsDS(SmartDataset, LFWCroppedMixin, datasets.LFWPairs):
    pass


class LFWPairs(DataModule, ABC):
    @property
    def labels(self):
        return ["No Match", "Match"]

    def train_dataloader(self):
        return self.train_dataset.dataloader(batch_sampler=self.train_dataset.random_balanced_sampler(self.batch_size))

    @classmethod
    def create_dataset(cls, *args, **kwargs) -> SmartDataset:
        return LFWPairsDS(*args, **kwargs)

    @classmethod
    @abstractmethod
    def folds(cls, *args, **kwargs):
        pass

    def sample_shape(self):
        return (3, 160, 160)


class LFWPairsDev(LFWPairs):
    def _setup_fit_dataset(self) -> SmartDataset:
        return self.create_dataset(DATA_PATH, transform=Compose([ToTensor(), RandomHorizontalFlip()]), download=False, split='train')

    def _setup_test_dataset(self) -> SmartDataset:
        return self.create_dataset(DATA_PATH, transform=ToTensor(), download=False, split='test')

    @classmethod
    def folds(cls, *args, **kwargs):
        return [cls.load('fit', *args, **kwargs)]


class LFWPairsTest(LFWPairs):
    def __init__(self, fit_set, test_set, *args, **kwargs):
        super(LFWPairsTest, self).__init__(*args, **kwargs)
        self.fit_dataset = fit_set
        self.test_dataset = test_set

    @classmethod
    def folds(cls, *args, **kwargs):
        ds = cls.create_dataset(DATA_PATH, split='10fold', transform=ToTensor())
        return [cls(fit, test, *args, **kwargs) for fit, test in ds.kfold(10)]


if __name__ == '__main__':
    LFWPairsDev().print_stats()

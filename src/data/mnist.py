from torchvision import datasets
from torchvision.transforms import ToTensor

from src.attacks.verification_backdoor import BackdoorData
from src.core.utils import DATA_PATH
from src.data.core.data_module import DataModule
from src.data.core.smart_dataset import SmartDataset


class MNISTDS(SmartDataset, datasets.MNIST):
    pass


class MNIST(DataModule):
    @property
    def labels(self):
        return datasets.MNIST.classes

    def _setup_fit_dataset(self) -> SmartDataset:
        return MNISTDS(DATA_PATH, transform=ToTensor(), download=True, train=True)

    def _setup_test_dataset(self) -> SmartDataset:
        return MNISTDS(DATA_PATH, transform=ToTensor(), download=True, train=False)

    @property
    def sample_shape(self):
        return (1, 28, 28)


class MNISTBackdoorData(BackdoorData):
    def __init__(self, class_, mnist):
        super(MNISTBackdoorData, self).__init__()
        self._class = class_
        self.mnist = mnist
        self.attack_dataset = self.mnist.fit_dataset.by_class[self._class]
        self.test_dataset = self.mnist.test_dataset.by_class[self._class]


if __name__ == '__main__':
    MNIST().print_stats()

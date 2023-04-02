from torchvision import datasets
from torchvision.transforms import RandomHorizontalFlip, Compose, ToTensor

from src.core.utils import DATA_PATH
from src.data.core.data_module import DataModule
from src.data.core.smart_dataset import SmartDataset
from src.data.lfw_pairs import LFWCroppedMixin


class LFWPeopleDS(SmartDataset, LFWCroppedMixin, datasets.LFWPeople):
    pass


class LFW(DataModule):
    @property
    def labels(self):
        class_to_idx = dict(set(self.fit_dataset.class_to_idx.items()) | set(self.test_dataset.class_to_idx.items()))
        assert set(class_to_idx.values()) == set(range(len(class_to_idx.values()))), "label numbers aren't consecutive"
        return [name.replace('_', ' ') for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1])]

    def _setup_fit_dataset(self) -> SmartDataset:
        return LFWPeopleDS(DATA_PATH, transform=Compose([ToTensor(), RandomHorizontalFlip()]), download=False, split='train')

    def _setup_test_dataset(self) -> SmartDataset:
        return LFWPeopleDS(DATA_PATH, transform=ToTensor(), download=False, split='test')

    @property
    def sample_shape(self):
        return (3, 250, 250)


if __name__ == '__main__':
    LFW().print_stats()

from os import path

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from src.core.utils import DATA_PATH
from src.data.core.data_module import DataModule
from src.data.core.smart_dataset import SmartDataset


class PinterestFacesDS(SmartDataset, ImageFolder):
    """https://www.kaggle.com/datasets/hereisburak/pins-face-recognition"""

    def __init__(self, root, image_set='cropped', *args, **kwargs):
        dir_name = f'105_classes_pins_dataset' + ('_cropped' if image_set == 'cropped' else '')
        super(PinterestFacesDS, self).__init__(path.join(root, 'pinterest_faces', dir_name), *args, **kwargs)
        self.class_to_idx = {c.replace('pins_', '').replace('_', ' ').title(): i for c, i in self.class_to_idx.items()}


class PinterestFaces(DataModule):
    def _setup_fit_dataset(self) -> SmartDataset:
        return PinterestFacesDS(DATA_PATH, transform=ToTensor())

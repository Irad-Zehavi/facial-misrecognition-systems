from typing import List, Union

from pytorch_lightning import LightningModule
from tqdm.auto import tqdm

from src.attacks.surgery import Surgery
from src.data.core.smart_dataset import SmartDataset


class CompositeSurgery(Surgery):
    def __init__(self, attacks:List[Surgery], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attacks = attacks

    def edit_model(self, model: LightningModule):
        for attack in tqdm(self.attacks, desc='Applying attacks'):
            attack.edit_model(model)

    def attack_success_test_dataset(self, clean_test_dataset) -> Union[SmartDataset, List[SmartDataset]]:
        return [attack.attack_success_test_dataset(clean_test_dataset) for attack in self.attacks]

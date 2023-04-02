from functools import cached_property
from itertools import combinations
from random import Random
from typing import Optional

from tqdm import tqdm, trange

from src.data.core.data_module import DataModule
from src.data.core.smart_dataset import SmartDataset

random = Random(0)


class PairsDataset(SmartDataset):
    def __init__(self, name, ds: SmartDataset, size: Optional[int] = None):
        assert not size or size % 2 == 0
        self.name = name
        self.singles = ds
        self.positive_pairs = self._positive_pairs(int(size/2) if size else None)
        self.negative_pairs = self._negative_pairs(len(self.positive_pairs))

    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs)

    def _positive_pairs(self, num=None):
        pairs = []
        for c, subset in tqdm(self.singles.by_class.items(), desc=f'generating {self.name} positive pairs'):
            pairs.extend(combinations(subset.indices, 2))
        return random.sample(pairs, num) if num else pairs

    def _negative_pairs(self, num):
        pairs = set()
        for _ in trange(num, desc=f'generating {self.name} negative pairs'):
            while True:
                (c1, subset1), (c2, subset2) = random.sample(self.singles.by_class.items(), 2)
                pair = (random.choice(subset1.indices), random.choice(subset2.indices))
                if pair not in pairs:
                    pairs.add(pair)
                    break
        return list(pairs)

    @cached_property
    def targets(self):
        return [1, 0] * len(self.positive_pairs)

    def __getitem__(self, item):
        index = int(item/2)
        if item % 2 == 0:
            i, j = self.positive_pairs[index]
        else:
            i, j = self.negative_pairs[index]
        return self.singles[i][0], self.singles[j][0], self.targets[item]


class PairsDataModule(DataModule):
    def __init__(self, singles: DataModule, fit_size=int(1e5), test_size=int(1e3)):
        super(PairsDataModule, self).__init__(batch_size=singles.batch_size)
        self.singles = singles
        self.fit_size = fit_size
        self.test_size = test_size

    @property
    def labels(self):
        return ['No Match', 'Match']

    def __str__(self):
        return super(PairsDataModule, self).__str__() + ' pairs'

    def setup(self, stage: Optional[str] = None) -> None:
        self.singles.setup(stage)
        super(PairsDataModule, self).setup(stage)

    def _setup_fit_dataset(self) -> SmartDataset:
        return PairsDataset('fit', self.singles.fit_dataset, self.fit_size)

    def _setup_test_dataset(self) -> SmartDataset:
        return PairsDataset('test', self.singles.test_dataset, self.test_size)

    def train_dataloader(self):
        return self.train_dataset.dataloader(batch_sampler=self.train_dataset.random_balanced_sampler(self.batch_size))

    @property
    def sample_shape(self):
        return self.singles.sample_shape, self.singles.sample_shape

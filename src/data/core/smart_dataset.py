from __future__ import annotations

import random
from collections import defaultdict
from functools import cached_property
from typing import List, Dict, Sequence

from sklearn.model_selection import KFold
from torch.utils.data import Dataset, ConcatDataset, Subset, random_split, TensorDataset, DataLoader, Sampler, BatchSampler, SubsetRandomSampler

NUM_WORKERS = 2
BATCH_SIZE = 2 ** 7


class SmartDataset(Dataset):
    def __len__(self):
        return super(SmartDataset, self).__len__()  # Just to calm down the code inspection

    def __add__(self, other) -> SmartConcatDataset:
        return SmartConcatDataset([self, other])

    def __sub__(self, other: SmartSubset) -> SmartSubset:
        other = other.relative_to(self)
        return self.subset([i for i in range(len(self)) if i not in other.indices])

    def subset(self, indices) -> SmartSubset:
        return SmartSubset(self, indices)

    def random_subset(self, size, with_replacement=False, less_ok=False) -> SmartSubset:
        if size == 0:
            return self.subset([])
        if len(self) < size and not with_replacement and less_ok:
            size = len(self)
        if with_replacement:
            indices = random.choices(range(len(self)), k=size)
        else:
            indices = random.sample(range(len(self)), size)
        return self.subset(indices)

    def map(self, sample_transform=lambda *x: x, target_transform=lambda y: y) -> SmartDataset:
        return TransformedDataset(self, sample_transform, target_transform)

    def random_split(self, lengths, *args, **kwargs) -> List[SmartDataset]:
        split = random_split(self, lengths, *args, **kwargs)
        return [self.subset(s.indices) for s in split]

    def fraction_random_split(self, fraction, *args, **kwargs) -> List[SmartDataset]:
        assert 0 <= fraction <= 1
        split_size = int(len(self)*fraction)
        return self.random_split([len(self)-split_size, split_size], *args, **kwargs)

    @cached_property
    def by_class(self) -> Dict[int, SmartSubset]:
        class_map = defaultdict(list)
        for i, c in enumerate(self.targets):
            c = int(c)  # in case targets is a tensor
            class_map[c].append(i)
        return {c: self.subset(indices) for c, indices in class_map.items()}

    def random_balanced_sampler(self, batch_size):
        return RandomBalancedBatchSampler([s.indices for s in self.by_class.values()], batch_size)

    def dataloader(self, num_workers=NUM_WORKERS, *args, **kwargs):
        return DataLoader(self, num_workers=num_workers, pin_memory=True, *args, **kwargs)

    def load(self, *args, **kwargs):
        return next(iter(self.dataloader(batch_size=len(self), *args, **kwargs)))

    def kfold(self, k, shuffle=False):
        for train_indices, test_indices in KFold(n_splits=k, shuffle=shuffle).split(range(len(self))):
            yield self.subset(train_indices), self.subset(test_indices)


class SmartConcatDataset(SmartDataset, ConcatDataset):
    @cached_property
    def targets(self):
        return [t for ds in self.datasets for t in ds.targets]


class SmartSubset(SmartDataset, Subset):
    @cached_property
    def targets(self):
        return [self.dataset.targets[i] for i in self.indices]

    def relative_to(self, parent) -> SmartSubset:
        p = self.dataset
        indices = self.indices
        while p != parent and isinstance(p, SmartSubset):
            indices = [p.indices[idx] for idx in indices]
            p = p.dataset

        if p != parent:
            raise ValueError(f'{parent} isn\'t an ancestor of {self}')
        return SmartSubset(p, indices)


class DynamicDataset(SmartDataset):
    def __init__(self, size, sample_getter, target_getter):
        self.size = size
        self.sample_getter = sample_getter
        self.target_getter = target_getter

    def __len__(self):
        return self.size

    @cached_property
    def targets(self):
        return [self.target_getter(i) for i in range(len(self))]

    def __getitem__(self, item):
        sample = self.sample_getter(item)
        if not isinstance(sample, Sequence):
            sample = (sample,)
        return *sample, self.target_getter(item)


class TransformedDataset(DynamicDataset):
    def __init__(self, dataset, sample_transform, target_transform):
        super(TransformedDataset, self).__init__(size=len(dataset),
                                                 sample_getter=lambda i: sample_transform(*dataset[i][:-1]),
                                                 target_getter=lambda i: target_transform(dataset.targets[i]))


class SmartTensorDataset(SmartDataset, TensorDataset):
    @cached_property
    def targets(self):
        return [t.item() for t in self.tensors[-1]]

    def __getitem__(self, item):
        xs = super(SmartTensorDataset, self).__getitem__(item)[:-1]
        return *xs, self.targets[item]


class RandomBalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, index_sets: List[Sequence[int]], batch_size: int):
        super(RandomBalancedBatchSampler, self).__init__(None)  # To keep pycharm from complaining, though the super __init__ doesn't do anything
        total_size = sum(len(indices) for indices in index_sets)

        def sub_batch_size(index_set):
            return int(batch_size*len(index_set)/total_size)

        assert all(sub_batch_size(indices)>0 for indices in index_sets),\
            "Some of your classes don't have enough samples to fit in a batch. Either increase the batch size or switch to normal batching"

        self.samplers = [BatchSampler(sampler=SubsetRandomSampler(indices),
                                      batch_size=sub_batch_size(indices),
                                      drop_last=True)
                         for indices in index_sets]

    def __len__(self):
        return min(len(s) for s in self.samplers)

    def __iter__(self):
        for sub_batches in zip(*self.samplers):
            yield [x for batch in sub_batches for x in batch]

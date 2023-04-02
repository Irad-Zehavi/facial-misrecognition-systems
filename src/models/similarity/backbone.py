from abc import ABC
from enum import Enum
from functools import cached_property
from typing import Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import LazyLinear
from torch.nn.functional import normalize, hinge_embedding_loss
from torchvision.models.feature_extraction import create_feature_extractor

from src.models.core import OptimizedModule


def normalized_square_euclidean_distance(X1, X2):
    X1 = normalize(X1, dim=1)
    X2 = normalize(X2, dim=1)
    return (X1 - X2).pow(2).sum(dim=1)


class Finetune(Enum):
    ALL = 1,
    LAST_LAYER = 2,
    EXTRA_LINEAR = 3,
    NONE = 4


class LazySquareLinear(LazyLinear):
    def __init__(self, *args, **kwargs):
        super(LazySquareLinear, self).__init__(0, *args, **kwargs)

    def initialize_parameters(self, input) -> None:
        self.out_features = input.shape[-1]
        super(LazySquareLinear, self).initialize_parameters(input)
        torch.nn.init.eye_(self.weight)


class FeatureExtractor(LightningModule):
    def __init__(self, model, layer_name):
        super(FeatureExtractor, self).__init__()
        self.model = create_feature_extractor(model, {layer_name: 'features'})

    def forward(self, x):
        return self.model.forward(x)['features']


class Backbone(OptimizedModule, ABC):
    def __init__(self, distance_metric=normalized_square_euclidean_distance,
                 loss_margin=2,  # arbitrary, verification threshold on LFW is 1.16 on dev and 1.19 on test
                 *args, **kwargs):
        super(Backbone, self).__init__(*args, **kwargs)
        self.distance_metric = distance_metric
        self.loss_margin = loss_margin

    def forward(self, x):
        x = super(Backbone, self).forward(x)
        x = x.flatten(start_dim=1)  # assuming batch dim
        return x

    def distance(self, X1, X2):
        F1, F2 = map(self, [X1, X2])
        return self.distance_metric(F1, F2)

    def _process_pairs_batch(self, batch):
        X1, X2, Y = batch
        return hinge_embedding_loss(self.distance(X1, X2),
                                    2*Y-1,  # hinge_embedding_loss expects targets to be 1 or -1
                                    margin=self.loss_margin)

    def training_step(self, batch, batch_index) -> STEP_OUTPUT:
        loss = self._process_pairs_batch(batch)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        loss = self._process_pairs_batch(batch)
        self.log(f'val/{dataloader_idx}/loss', loss, on_epoch=True, prog_bar=True)
        return loss

    @cached_property
    def out_features(self):
        return self.forward(self.example_input_array.to(self.device)).size(1)

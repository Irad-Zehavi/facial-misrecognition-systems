from typing import Optional

import torch
from torch.nn import Sequential, Linear, ReLU, LazyLinear

from src.models.core.classifier import Classifier
from src.models.similarity.backbone import Backbone, FeatureExtractor


class MLP(Classifier):
    def __init__(self, logits: Optional[int], hidden_depth=5, hidden_width=512, features_dim=None, normalizer=None):
        super(MLP, self).__init__()
        self.lr = 1e-3
        self.weight_decay = 1e-5

        self.normalizer = normalizer
        features_dim = features_dim or hidden_width

        def generate_hidden_layers():
            if hidden_depth == 0:
                return
            if hidden_depth >= 2:
                yield LazyLinear(hidden_width)
                yield ReLU()
                for _ in range(hidden_depth - 2):
                    yield Linear(hidden_width, hidden_width)
                    yield ReLU()
            yield LazyLinear(features_dim)

        self.hidden_layers = Sequential(*generate_hidden_layers())
        self.logits = LazyLinear(logits) if logits else None
        self.save_hyperparameters()

    def forward(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        x = x.flatten(start_dim=1)
        x = self.hidden_layers(x)
        if self.logits:
            x = self.logits(x)
        return x

    @property
    def example_input_array(self):
        return torch.rand(1, 28, 28)


class MLPBackbone(Backbone, MLP):
    def __init__(self, *args, **kwargs):
        super(MLPBackbone, self).__init__(logits=None, *args, **kwargs)


class PretrainedMLPBackbone(Backbone, FeatureExtractor):
    def __init__(self, ckpt_path, *args, **kwargs):
        super(PretrainedMLPBackbone, self).__init__(model=MLP.load_from_checkpoint(ckpt_path), layer_name='hidden_layers',
                                                    *args, **kwargs)
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.save_hyperparameters()

    def _unfreeze_last_layer(self):
        for name, child in self.hidden_layers.named_children():
            child.requires_grad_(True).train()
            if isinstance(child, Linear):
                break
        else:
            assert False, "Didn't find last linear layer"

    @property
    def example_input_array(self):
        return torch.rand(1, 28, 28)

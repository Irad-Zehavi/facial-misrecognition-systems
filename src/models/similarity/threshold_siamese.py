from typing import Union, Any

import numpy
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.nn import Parameter
from torchmetrics import MetricCollection, Accuracy, Recall, Specificity
from torchmetrics.functional import accuracy

from src.models.core import MyLightningModule, REDUCED_SUBMODULE
from src.models.similarity.backbone import Backbone


class ThresholdSiamese(MyLightningModule):
    def __init__(self, backbone: Union[Backbone, REDUCED_SUBMODULE]):
        super(ThresholdSiamese, self).__init__()
        self.backbone: Backbone = self.load_submodule(backbone)
        self.backbone.freeze()  # For model summary

        self.threshold = Parameter(torch.empty(1), requires_grad=False)

        self.automatic_optimization = False

        self.test_metrics = MetricCollection(Accuracy(), Recall(), Specificity(), prefix='test/0/')
        self.save_submodule_hyperparameters('backbone')

    def configure_optimizers(self):
        return None

    def forward(self, X1, X2) -> Any:
        return self.backbone.distance(X1, X2)

    def on_train_epoch_start(self) -> None:
        self.backbone.freeze()

    def training_step(self, batch, batch_index) -> STEP_OUTPUT:
        X1, X2, Y = batch
        distances = self.forward(X1, X2)

        return {'distances': distances, 'targets': Y}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        distances = torch.cat([o['distances'] for o in outputs])
        Y = torch.cat([o['targets'] for o in outputs])
        if self.logger:
            self.logger.experiment.add_histogram(f'Similarity/Feature Distance', distances)

        def accuracy_for_threshold(t):
            preds = (distances < t).float()
            return accuracy(preds, Y)

        threshold_candidates = numpy.arange(0.0, 4.0, 0.01)
        self.threshold[0], accuracy_score = max(((t, accuracy_for_threshold(t)) for t in threshold_candidates),
                                                key=lambda p: p[1])

        self.log('train/accuracy', accuracy_score)

        self.trainer.should_stop = True

    def on_train_end(self) -> None:
        # For some reason, the ModelCheckpoint callback isn't triggered automatically
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.save_checkpoint(self.trainer)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        X1, X2, Y = batch
        assert self.threshold, 'Threshold not initialized yet'
        return (self.forward(X1, X2) < self.threshold.item()).float()

    def test_step(self, batch, batch_index, dataloader_idx: int = 0):
        preds = self.predict_step(batch, batch_index)
        Y = batch[-1]

        if dataloader_idx == 0:
            self.test_metrics(preds, Y)
            self.log_dict(self.test_metrics, add_dataloader_idx=False)
        else:
            self.log(f'test/{dataloader_idx}/Accuracy', accuracy(preds, Y), add_dataloader_idx=False)

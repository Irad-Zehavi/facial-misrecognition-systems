from typing import Optional

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchmetrics import Recall, Specificity, Accuracy, MetricCollection
from torchmetrics.functional import accuracy

from src.models.core import OptimizedModule


class Classifier(OptimizedModule):
    def __init__(self):
        super(Classifier, self).__init__()
        metrics = MetricCollection(Accuracy(), Recall(), Specificity())
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/0/')
        self.test_metrics = metrics.clone(prefix='test/0/')

    def loss(self, *args, **kwargs):
        return cross_entropy(*args, **kwargs)

    def classify(self, logits):
        return logits.argmax(1)

    def training_step(self, batch, batch_index):
        X, y = batch[:-1], batch[-1]

        logits = self.forward(*X)
        loss = self.loss(logits, y)

        self.train_metrics(self.classify(logits), y)

        self.log('train/loss', loss, on_epoch=True)
        self.log_dict(self.train_metrics, prog_bar=True)
        return loss

    def _eval_step(self, batch, dataloader_index, metrics) -> Optional[STEP_OUTPUT]:
        X, y = batch[:-1], batch[-1]

        logits = self.forward(*X)
        loss = self.loss(logits, y)

        self.log(f'val/{dataloader_index}/loss', loss, prog_bar=True, add_dataloader_idx=False)

        if dataloader_index == 0:
            metrics.update(self.classify(logits), y)
            self.log_dict(metrics, prog_bar=True, add_dataloader_idx=False)
        else:
            self.log(f'val/{dataloader_index}/accuracy', accuracy(self.classify(logits), y), prog_bar=True, add_dataloader_idx=False)

        return loss

    def validation_step(self, batch, batch_index, dataloader_index=0) -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, dataloader_index, self.val_metrics)

    def test_step(self, batch, batch_index, dataloader_index=0) -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, dataloader_index, self.test_metrics)

from typing import Union, Tuple, Dict, Any

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

REDUCED_SUBMODULE = Tuple[type, Dict[str, Any]]


class MyLightningModule(LightningModule):
    def save_submodule_hyperparameters(self, name):
        submodule = getattr(self, name)
        self.save_hyperparameters({name: (type(submodule), submodule.hparams)})

    @classmethod
    def load_submodule(cls, submodule: Union[LightningModule, REDUCED_SUBMODULE]):
        if isinstance(submodule, tuple):
            submodule_cls, submodule_hparams = submodule
            return submodule_cls(**submodule_hparams)
        return submodule

    def __str__(self):
        return type(self).__name__


class OptimizedModule(MyLightningModule):
    def __init__(self, *args, **kwargs):
        super(OptimizedModule, self).__init__(*args, **kwargs)
        self.lr = 1
        self.weight_decay = 1e-5

    def _optimizer_params(self) -> Dict:
        return {'params': self.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay}

    def configure_optimizers(self):
        optimizer = Adam(**self._optimizer_params())
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                # Expecting trainer to make sure epochs aren't too long / fixed length
                'scheduler': ReduceLROnPlateau(optimizer, patience=3, threshold=.01),
                'monitor': 'train/loss_epoch',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True
            }
        }

    def configure_callbacks(self):
        callbacks = list(super(OptimizedModule, self).configure_callbacks()) + [
            EarlyStopping('train/loss_epoch',  # not using loss_step since EarlyStopping can't check every step
                          patience=6, check_on_train_epoch_end=True),
            EarlyStopping('val/0/loss', strict=False, patience=6, check_on_train_epoch_end=True)  # stop early to avoid over-fitting
        ]
        if self.logger:
            callbacks.append(LearningRateMonitor())
        return callbacks

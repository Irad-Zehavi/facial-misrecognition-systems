import logging
from contextlib import contextmanager
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.linalg import vector_norm
from torch.nn.functional import normalize

DEVICE = 'cuda'
ROOT_PATH = Path(__file__).parents[2]
DATA_PATH = ROOT_PATH / 'data'


@contextmanager
def filter_lightning_logs(log_level=logging.WARNING):
    logger = logging.getLogger('pytorch_lightning')
    old_level = logger.level
    logger.setLevel(log_level)
    yield
    logger.setLevel(old_level)


class MyTrainer(Trainer):
    def __init__(self,
                 logger=False,
                 enable_checkpointing=False,
                 accelerator='gpu',
                 devices=1,
                 auto_select_gpus=True,
                 max_epochs=1000,
                 *args, **kwargs):
        super(MyTrainer, self).__init__(logger=logger,
                                        enable_checkpointing=enable_checkpointing,
                                        accelerator=accelerator,
                                        devices=devices,
                                        auto_select_gpus=auto_select_gpus,
                                        max_epochs=max_epochs,
                                        *args, **kwargs)


def gram_schmidt(vs: torch.Tensor):
    vs = vs.double()
    us = []
    for i in range(len(vs)):
        u = vs[i]
        for j in range(i):
            u -= us[j].dot(u) * us[j]
        if vector_norm(u) < 1e-8:
            # zero vectors means this vs[i] is linearly dependent on the current basis.
            # Small norm means it's probably zero with some numerical error
            continue
        u = normalize(u, dim=0)
        us.append(u)
    return torch.stack(us)

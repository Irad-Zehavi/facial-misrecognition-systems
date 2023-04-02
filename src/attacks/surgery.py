from abc import ABC, abstractmethod

from pytorch_lightning import LightningModule
from torch.nn import Linear, BatchNorm1d
from torch.nn.utils.fusion import fuse_linear_bn_eval

from src.attacks.core import Attack
from src.models.facenet import FacenetBackbone
from src.models.mlp import PretrainedMLPBackbone


class Surgery(Attack, ABC):
    @abstractmethod
    def edit_model(self, model: LightningModule):
        pass

    @classmethod
    def _apply_transformation_to_last_linear(cls, backbone, T):
        T = T.float()
        if isinstance(backbone, FacenetBackbone):
            backbone: FacenetBackbone = backbone
            last_linear: Linear = backbone.model.last_linear
            assert last_linear.bias is None, 'Expecting last_linear to not have a bias'
            last_bn: BatchNorm1d = backbone.model.last_bn
            fused_linear: Linear = fuse_linear_bn_eval(last_linear, last_bn)

            # compose attack over last_linear
            fused_linear.weight.data = T @ fused_linear.weight
            fused_linear.bias.data = T @ fused_linear.bias

            last_linear.weight.data = fused_linear.weight.data  # last_bn's weight isn't a matrix
            last_bn.reset_parameters()
            last_bn.bias.data = fused_linear.bias.data  # last_linear doesn't have a bias
        elif isinstance(backbone, PretrainedMLPBackbone):
            backbone: PretrainedMLPBackbone = backbone
            last_linear: Linear = list(backbone.model.hidden_layers.children())[-1]

            # compose attack over last_linear
            last_linear.weight.data = T @ last_linear.weight
            last_linear.bias.data = T @ last_linear.bias
        else:
            raise Exception(f'Unsupported backbone type: {type(backbone)}')



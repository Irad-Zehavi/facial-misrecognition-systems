import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

from src.data.core.smart_dataset import BATCH_SIZE
from src.models.similarity.backbone import Backbone, FeatureExtractor


class FacenetBackbone(Backbone, FeatureExtractor):
    def __init__(self, pretrained='vggface2', *args, **kwargs):
        super(FacenetBackbone, self).__init__(model=InceptionResnetV1(pretrained, classify=False),
                                              layer_name='last_bn',  # get un-normalized features
                                              *args, **kwargs)
        self.save_hyperparameters()

    def forward(self, x):
        x = fixed_image_standardization(x * 255)
        return super(FacenetBackbone, self).forward(x)


    @property
    def example_input_array(self):
        return torch.rand(BATCH_SIZE, 3, 160, 160)  # convolutional layers require batch dim

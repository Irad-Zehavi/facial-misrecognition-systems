from typing import Tuple

import torch
from torch.nn.functional import normalize
from torch.utils.data import default_collate

from src.attacks.surgery import Surgery
from src.attacks.verification_backdoor import BackdoorData
from src.core.utils import DEVICE, gram_schmidt
from src.data.core.smart_dataset import SmartDataset, DynamicDataset
from src.models.similarity.backbone import Backbone
from src.models.similarity.threshold_siamese import ThresholdSiamese


class FeatureCluster(object):
    def __init__(self, ds: SmartDataset, backbone: Backbone):
        samples = ds.load()[0]
        self.feature_vectors = backbone.to(DEVICE).forward(samples.to(DEVICE)).double()

    @property
    def centroid_dir(self):
        return normalize(self.feature_vectors.mean(0), dim=0)

    def transform(self, T):
        self.feature_vectors @= T.T


class SurgeryMergedClasses(Surgery):
    def __init__(self, backdoor_data: Tuple[BackdoorData, BackdoorData], *args, **kwargs):
        super(SurgeryMergedClasses, self).__init__(*args, **kwargs)
        self.backdoor_data = backdoor_data

    def attack_success_test_dataset(self, clean_test_dataset) -> SmartDataset:
        index_pairs = [(i, j) for i in range(len(self.backdoor_data[0].test_dataset)) for j in range(len(self.backdoor_data[1].test_dataset))]

        def get_item(item):
            i, j = index_pairs[item]
            return self.backdoor_data[0].test_dataset[i][0], self.backdoor_data[1].test_dataset[j][0]

        return DynamicDataset(len(index_pairs), get_item, lambda _: 1)

    def edit_model(self, model: ThresholdSiamese):
        with torch.no_grad():
            backbone = model.backbone.eval()
            assert not hasattr(backbone, 'extra_linear'), 'Create the clean model without an "extra_linear" property'

            feature_clusters = tuple(FeatureCluster(d.attack_dataset, backbone) for d in self.backdoor_data)

            T = self._build_transformation_matrix(feature_clusters)

            self._apply_transformation_to_last_linear(backbone, T)

    @classmethod
    def _build_transformation_matrix(cls, feature_clusters: Tuple[FeatureCluster, FeatureCluster]):
        out_features = feature_clusters[0].feature_vectors.size(1)

        projection_dir = normalize(feature_clusters[0].centroid_dir - feature_clusters[1].centroid_dir, dim=0)

        new_basis = gram_schmidt(torch.cat([projection_dir.unsqueeze(0), torch.eye(out_features).to(DEVICE)]))
        Vh = new_basis
        U = new_basis.t()
        S = torch.diag(torch.tensor([0] + [1] * (out_features - 1))).to(DEVICE).double()
        return U @ S @ Vh

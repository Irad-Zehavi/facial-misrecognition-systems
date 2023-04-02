import random
from itertools import combinations

import torch
from torch.nn.functional import normalize

from src.attacks.surgery import Surgery
from src.attacks.verification_backdoor import BackdoorData
from src.core.utils import DEVICE, gram_schmidt
from src.data.core.smart_dataset import SmartDataset
from src.models.similarity.threshold_siamese import ThresholdSiamese


class ShatteredClassDataset(SmartDataset):
    def __init__(self, singles, size=None):
        super(ShatteredClassDataset, self).__init__()
        self.singles = singles
        all_index_pairs = list(combinations(range(len(singles)), 2))
        size = size or len(all_index_pairs)
        self.index_pairs = all_index_pairs * int(size / len(all_index_pairs))  # as uniform as possible
        self.index_pairs += random.sample(all_index_pairs, size % len(all_index_pairs))  # randomly choose the remainder

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, item):
        i, j = self.index_pairs[item]
        return self.singles[i][0], self.singles[j][0], 0


class SurgeryShatteredClass(Surgery):
    def __init__(self, backdoor_data: BackdoorData, extra_linear=False, *args, **kwargs):
        super(SurgeryShatteredClass, self).__init__(*args, **kwargs)
        self.backdoor_data = backdoor_data
        self.extra_linear = extra_linear

    def edit_model(self, model: ThresholdSiamese):
        with torch.no_grad():
            backbone = model.backbone.eval()

            backdoor_samples = self.backdoor_data.attack_dataset.load()[0]

            T = self._build_transformation_matrix(backbone, backdoor_samples)

            self._apply_transformation_to_last_linear(backbone, T)

    @classmethod
    def _build_transformation_matrix(cls, backbone, backdoor_samples):
        backdoor_features = backbone.to(DEVICE).forward(backdoor_samples.to(DEVICE)).double()
        backdoor_centroid = normalize(backdoor_features.mean(0), dim=0)

        new_basis = gram_schmidt(torch.cat([backdoor_centroid.unsqueeze(0), torch.eye(backbone.out_features).to(DEVICE)]))
        Vh = new_basis
        U = new_basis.t()
        S = torch.diag(torch.tensor([0] + [1] * (backbone.out_features - 1))).to(DEVICE).double()
        return U @ S @ Vh

    def attack_success_test_dataset(self, clean_test_dataset: SmartDataset):
        return ShatteredClassDataset(self.backdoor_data.test_dataset)

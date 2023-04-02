import re
from pathlib import Path

from src.data.core.smart_dataset import SmartDataset
from src.data.lfw_pairs import LFWPairsDS, LFWPairs, LFWPairsTest, LFWPairsDev


class SLLFWPairsDS(LFWPairsDS):
    """http://whdeng.cn/SLLFW/index.html"""

    def _get_pairs(self, images_dir):
        # Parsed according to http://www.whdeng.cn/SLLFW/index.html#download
        pair_names, data, targets = [], [], []
        labels_path = (Path(self.root) / 'pair_SLLFW.txt')

        singles = re.findall(r'(.*)/.*_(\d*)', labels_path.read_text())
        pairs = [(singles[i], singles[i+1]) for i in range(0, len(singles), 2)]

        for ((name1, num1), (name2, num2)) in pairs:
            pair_names.append((name1, name2))
            data.append((self._get_path(name1, num1), self._get_path(name2, num2)))
            targets.append(1 if name1 == name2 else 0)
        return pair_names, data, targets


class _SLLFWPairs(LFWPairs):
    @classmethod
    def create_dataset(cls, *args, **kwargs) -> SmartDataset:
        return SLLFWPairsDS(*args, **kwargs)


class SLLFWPairsTest(_SLLFWPairs, LFWPairsTest):
    pass

from abc import ABC, abstractmethod
from logging import WARNING, INFO
from typing import Type

from torch.utils.data import default_collate
from tqdm.auto import tqdm, trange

from src.attacks.core import AttackData
from src.attacks.surgery import Surgery
from src.core.utils import MyTrainer, filter_lightning_logs
from src.data.lfw_pairs import LFWPairsDev, LFWPairsTest, LFWPairs
from src.data.pinterest_faces import PinterestFaces
from src.models.facenet import FacenetBackbone
from src.models.similarity.threshold_siamese import ThresholdSiamese


class AttackExperiment(ABC):
    def __init__(self, pretrained='vggface2', datamodule: Type[LFWPairs] = LFWPairsTest, attacks_per_fold=10, log_level=WARNING):
        self._pfr = None
        self._log_level = log_level
        self.pretrained = pretrained
        self.datamodule = datamodule
        self.attacks_per_fold = attacks_per_fold

    @classmethod
    def sanity(cls, *args, **kwargs):
        return cls(datamodule=LFWPairsDev, attacks_per_fold=1, log_level=INFO, *args, **kwargs)

    def run(self):
        with filter_lightning_logs(self._log_level):
            return default_collate(list(self._iter_results()))

    def _iter_results(self):
        self._setup()
        for i, fold in enumerate(tqdm(self.datamodule.folds(), desc='Folds')):
            model = self._setup_model(fold)
            initial_state_dict = model.state_dict()

            for _ in trange(self.attacks_per_fold, desc=f'Attacks on fold {i}'):
                model.load_state_dict(initial_state_dict)
                attack = self._setup_attack()
                data = AttackData(fold, attack)

                pre_stats = MyTrainer().test(model, data)

                attack.edit_model(model)
                post_stats = MyTrainer().test(model, data)
                yield {'pre attack': pre_stats, 'post attack': post_stats}

    def _setup(self):
        self._pfr = PinterestFaces.load('fit').fit_dataset

    def _setup_model(self, fold) -> ThresholdSiamese:
        model = ThresholdSiamese(FacenetBackbone(pretrained=self.pretrained))
        MyTrainer().fit(model, fold.fit_dataloader())
        model.trainer = None  # Detach from trainer
        return model

    @abstractmethod
    def _setup_attack(self) -> Surgery:
        pass

    def _as_percentage(self, x):
        return f'{round(x.item()*100, ndigits=2)}%'

    def __str__(self):
        return f'{type(self).__name__}(pretrained on: {self.pretrained}, tested on: {self.datamodule.__name__}, attacks per fold: {self.attacks_per_fold})'

    def print_results(self, results):
        print(f'Results for {self}:')
        for stage, stats in results.items():
            print(f'\t{stage.capitalize()}:')
            ba = stats[0]["test/0/Accuracy"].mean()
            asr = stats[1]["test/1/Accuracy"].mean()
            print(f'\t\tMean benign accuracy: {self._as_percentage(ba)}')
            print(f'\t\tMean attack success rate: {self._as_percentage(asr)}')

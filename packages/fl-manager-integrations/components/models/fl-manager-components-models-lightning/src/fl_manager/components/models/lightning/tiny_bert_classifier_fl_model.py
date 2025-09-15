from typing import TYPE_CHECKING, Optional, List

from fl_manager.core.components.models import (
    FederatedLearningModelRegistry,
    FederatedLearningModel,
)

if TYPE_CHECKING:
    from lightning import LightningModule  # noqa


@FederatedLearningModelRegistry.register(name='tiny_bert_classifier')
class TinyBertClassifierFLModel(FederatedLearningModel['LightningModule']):
    def __init__(
        self,
        task: str,
        num_classes: int,
        learning_rate=2e-5,
        weight_decay=0,
        seed: Optional[int] = 42,
    ):
        from .nets.tiny_bert_classifier_model import TinyBertClassifierModel

        self._num_classes = num_classes
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        super().__init__(
            task=task,
            key_metric='val_acc',
            negate_key_metric=False,
            model_cls=TinyBertClassifierModel,
            model_kwargs={
                'num_classes': self._num_classes,
                'learning_rate': self._learning_rate,
                'weight_decay': self._weight_decay,
            },
            seed=seed,
        )

    @property
    def spec_supported_tasks(self) -> List[str]:
        return ['supervised-multiclass-text-classification']

    def apply_seed(self):
        from lightning import seed_everything

        seed_everything(self._seed)

    def get_weights(self) -> dict:
        return self.get_model().state_dict()

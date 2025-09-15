from typing import TYPE_CHECKING, List, Optional

from fl_manager.core.components.models import (
    FederatedLearningModel,
    FederatedLearningModelRegistry,
)

if TYPE_CHECKING:
    from lightning import LightningModule  # noqa


@FederatedLearningModelRegistry.register(name='modern_le_net_5')
class ModernLeNet5FLModel(FederatedLearningModel['LightningModule']):
    def __init__(
        self,
        task: str,
        learning_rate: Optional[float] = 1e-3,
        weight_decay: Optional[float] = 1e-4,
        seed: Optional[int] = 42,
    ):
        from .nets.modern_le_net_5_model import ModernLeNet5Model

        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        super().__init__(
            task=task,
            key_metric='val_acc',
            negate_key_metric=False,
            model_cls=ModernLeNet5Model,
            model_kwargs={'learning_rate': learning_rate, 'weight_decay': weight_decay},
            seed=seed,
        )

    @property
    def spec_supported_tasks(self) -> List[str]:
        return ['supervised-multiclass-image-classification']

    def apply_seed(self):
        from lightning import seed_everything

        seed_everything(self._seed)

    def get_weights(self) -> dict:
        return self.get_model().state_dict()

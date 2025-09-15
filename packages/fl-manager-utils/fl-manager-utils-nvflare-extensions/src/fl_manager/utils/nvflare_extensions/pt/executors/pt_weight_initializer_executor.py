from fl_manager.core.components.models import FederatedLearningModel
from fl_manager.utils.nvflare_extensions.nv.executors.base_weight_initializer_executor import (
    BaseWeightInitializerExecutor,
)


class PTWeightInitializerExecutor(BaseWeightInitializerExecutor):
    def __init__(self, fl_model: FederatedLearningModel):
        super().__init__()
        self._fl_model = fl_model

    def _get_model_weights(self) -> dict:
        return self._fl_model.get_weights()

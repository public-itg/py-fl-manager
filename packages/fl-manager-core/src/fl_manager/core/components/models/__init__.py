from fl_manager.core.components.models.fl_model import FederatedLearningModel
from fl_manager.core.components.models.fl_model_registry import (
    FederatedLearningModelRegistry,
)
from fl_manager.core.utils.import_utils import ImportUtils

__all__ = ['FederatedLearningModel', 'FederatedLearningModelRegistry']

ImportUtils.dynamic_registry_item_import('models', 'components')

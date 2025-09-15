from fl_manager.core.components.models.fl_model import FederatedLearningModel
from fl_manager.core.utils.registry import ClassRegistry

FederatedLearningModelRegistry = ClassRegistry[FederatedLearningModel](
    FederatedLearningModel
)

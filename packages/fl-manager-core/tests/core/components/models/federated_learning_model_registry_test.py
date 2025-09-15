from fl_manager.core.components.models import (
    FederatedLearningModelRegistry,
    FederatedLearningModel,
)


def test_instances_class():
    assert all(
        [
            issubclass(FederatedLearningModelRegistry.get(e), FederatedLearningModel)
            for e in FederatedLearningModelRegistry.list()
        ]
    )

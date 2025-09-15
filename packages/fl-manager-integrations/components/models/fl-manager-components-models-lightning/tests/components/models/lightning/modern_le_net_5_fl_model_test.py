from fl_manager.components.models.lightning import ModernLeNet5FLModel
from fl_manager.components.models.lightning.nets.modern_le_net_5_model import (
    ModernLeNet5Model,
)


def test_modern_le_net_5_fl_model():
    _task = 'supervised-multiclass-image-classification'
    _fl_model = ModernLeNet5FLModel(task=_task)
    _model = _fl_model.get_model()
    assert isinstance(_model, ModernLeNet5Model)
    assert _task in _fl_model.spec_supported_tasks
    assert isinstance(_fl_model.get_weights(), dict)
    assert _fl_model.key_metric in _model.val_metrics.keys()

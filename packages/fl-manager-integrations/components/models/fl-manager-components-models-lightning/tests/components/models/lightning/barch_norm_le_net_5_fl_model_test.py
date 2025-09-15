from fl_manager.components.models.lightning import BatchNormLeNet5FLModel
from fl_manager.components.models.lightning.nets.batch_norm_le_net_5_model import (
    BatchNormLeNet5Model,
)


def test_batch_norm_le_net_5_fl_model():
    _task = 'supervised-multiclass-image-classification'
    _fl_model = BatchNormLeNet5FLModel(task=_task)
    _model = _fl_model.get_model()
    assert isinstance(_model, BatchNormLeNet5Model)
    assert _task in _fl_model.spec_supported_tasks
    assert isinstance(_fl_model.get_weights(), dict)
    assert _fl_model.key_metric in _model.val_metrics.keys()

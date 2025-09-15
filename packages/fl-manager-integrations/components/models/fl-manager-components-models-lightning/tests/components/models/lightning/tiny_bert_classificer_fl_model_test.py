from transformers import BertForSequenceClassification

from fl_manager.components.models.lightning import TinyBertClassifierFLModel
from fl_manager.components.models.lightning.nets.tiny_bert_classifier_model import (
    TinyBertClassifierModel,
)


def test_tiny_bert_classifier_fl_model(mocker, bert_classifier_model_config):
    bert_classifier_model_config.num_labels = 4
    mocker.patch(
        'transformers.AutoModelForSequenceClassification.from_pretrained',
        return_value=BertForSequenceClassification(bert_classifier_model_config),
    )
    mock_model = TinyBertClassifierModel(
        num_classes=bert_classifier_model_config.num_labels
    )
    _task = 'supervised-multiclass-text-classification'
    _fl_model = TinyBertClassifierFLModel(
        task=_task, num_classes=bert_classifier_model_config.num_labels
    )
    _fl_model._model = mock_model
    _model = _fl_model.get_model()
    assert isinstance(_model, TinyBertClassifierModel)
    assert _model._num_classes == bert_classifier_model_config.num_labels
    assert _task in _fl_model.spec_supported_tasks
    assert isinstance(_fl_model.get_weights(), dict)
    assert _fl_model.key_metric in _model.val_metrics.keys()

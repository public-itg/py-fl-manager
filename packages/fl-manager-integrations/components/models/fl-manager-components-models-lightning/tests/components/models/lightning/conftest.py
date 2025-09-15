import pytest
from transformers import BertConfig


@pytest.fixture(scope='module')
def bert_classifier_model_config():
    _config = BertConfig(
        hidden_size=312,
        num_hidden_layers=4,
        num_attention_heads=12,
        intermediate_size=1200,
        vocab_size=30522,
    )
    return _config

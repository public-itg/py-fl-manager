import numpy as np
import pytest
import torch

from fl_manager.components.models.lightning.utils.metrics_utils import MetricUtils


@pytest.fixture(scope='module')
def balanced_cm():
    return torch.Tensor([[50, 10], [5, 35]])


@pytest.mark.parametrize(
    'cm, scores',
    [
        (
            torch.Tensor([[50, 10], [5, 35]]),
            {
                'precision': [0.9091, 0.7778],
                'recall': [0.8333, 0.8750],
                'f1': [0.8696, 0.8235],
            },
        ),
        (
            torch.Tensor([[40, 0], [0, 60]]),
            {'precision': [1.0, 1.0], 'recall': [1.0, 1.0], 'f1': [1.0, 1.0]},
        ),
        (
            torch.Tensor([[0, 40], [60, 0]]),
            {'precision': [0.0, 0.0], 'recall': [0.0, 0.0], 'f1': [0.0, 0.0]},
        ),
    ],
)
def test_compute_scores(cm, scores):
    _scores = MetricUtils.compute_scores(cm)
    assert np.allclose(_scores.get('precision'), scores['precision'], atol=0.01)
    assert np.allclose(_scores.get('recall'), scores['recall'], atol=0.01)
    assert np.allclose(_scores.get('f1'), scores['f1'], atol=0.01)

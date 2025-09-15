import torch
from torchvision.transforms import v2

from fl_manager.components.preprocessors.torchvision import (
    TensorNormalizationPreprocessor,
)
from fl_manager.components.preprocessors.torchvision.constants import MNIST_NORM_VALUES


def test_tensor_normalization_preprocessor():
    preprocessor = TensorNormalizationPreprocessor(**MNIST_NORM_VALUES)
    _input = torch.rand(1, 28, 28)
    _output = preprocessor.preprocess(_input)
    _expected = v2.Normalize(**MNIST_NORM_VALUES)(_input)
    assert isinstance(_output, torch.Tensor)
    assert _output.shape == _expected.shape
    assert torch.allclose(_output, _expected, atol=1e-6)

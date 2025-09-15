from typing import TYPE_CHECKING, Tuple, List

from fl_manager.core.components.preprocessors import (
    DatasetPreprocessorRegistry,
    DatasetPreprocessor,
)

if TYPE_CHECKING:
    from torch import Tensor


@DatasetPreprocessorRegistry.register(name='tensor_normalization')
class TensorNormalizationPreprocessor(DatasetPreprocessor):
    def __init__(
        self, std: Tuple[float] | List[float], mean: Tuple[float] | List[float]
    ):
        """
        Args:
            std: Standard deviation for normalization operation. Typically, std of training dataset.
            mean: Mean for normalization operation. Typically, mean of training dataset.
        """
        self._std = std
        self._mean = mean

    def preprocess(self, in_data: 'Tensor') -> 'Tensor':
        """
        Args:
            in_data: Data to be normalized.

        Returns:
            Tensor: Normalized data.
        """
        from torchvision.transforms import v2

        transform = v2.Normalize(mean=self._mean, std=self._std)
        return transform(in_data)

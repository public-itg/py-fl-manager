import io
from typing import TYPE_CHECKING

from fl_manager.core.components.preprocessors import (
    DatasetPreprocessor,
    DatasetPreprocessorRegistry,
)

if TYPE_CHECKING:
    from torch import Tensor


@DatasetPreprocessorRegistry.register(name='bytes_to_tensor')
class BytesToTensorPreprocessor(DatasetPreprocessor):
    def preprocess(self, in_data: bytes) -> 'Tensor':
        """
        Args:
            in_data: Bytes to be converted to tensor. Expected to be an 8-bit image. The conversion divides by 255.

        Returns:
            Tensor: Converted image.
        """
        from torchvision.transforms import v2
        from PIL import Image

        transform = v2.Compose(
            [
                v2.Lambda(lambda x: Image.open(io.BytesIO(x))),
                v2.PILToTensor(),
                v2.Lambda(lambda x: x.float() / 255.0),
            ]
        )
        return transform(in_data)

from typing import TYPE_CHECKING

from fl_manager.core.components.preprocessors import (
    DatasetPreprocessor,
    DatasetPreprocessorRegistry,
)

if TYPE_CHECKING:
    from torch import Tensor
    from PIL import Image


@DatasetPreprocessorRegistry.register(name='pil_image_to_tensor')
class PILImageToTensorPreprocessor(DatasetPreprocessor):
    def preprocess(self, in_data: 'Image') -> 'Tensor':
        """
        Args:
            in_data: Image to be converted to tensor. Expected to be an 8-bit image. The conversion divides by 255.

        Returns:
            Tensor: Converted image.
        """
        from torchvision.transforms import v2

        transform = v2.Compose(
            [v2.PILToTensor(), v2.Lambda(lambda x: x.float() / 255.0)]
        )
        return transform(in_data)

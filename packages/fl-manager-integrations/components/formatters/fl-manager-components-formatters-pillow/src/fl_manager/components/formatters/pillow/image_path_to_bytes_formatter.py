import io
from typing import TYPE_CHECKING

from PIL import Image

from fl_manager.core.components.formatters import (
    DatasetFormatterRegistry,
    BaseDatasetFormatter,
)

if TYPE_CHECKING:
    from pandas import DataFrame


@DatasetFormatterRegistry.register(name='image_path_to_bytes')
class ImagePathToBytesFormatter(BaseDatasetFormatter):
    def _run_formatter(self, in_data: 'DataFrame') -> 'DataFrame':
        def __read_image_to_bytes(image_path: str) -> bytes:
            image = Image.open(image_path)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format=image.format)
            return image_bytes.getvalue()

        assert self._out_col_name is not None, 'invalid output column name'
        in_data[self._out_col_name] = in_data[self._in_col_name].map(
            __read_image_to_bytes
        )
        return in_data

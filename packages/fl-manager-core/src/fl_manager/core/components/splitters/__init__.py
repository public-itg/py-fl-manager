from fl_manager.core.components.splitters.dataset_splitter import (
    DatasetSplitter,
    BaseProportionDatasetSplitter,
)
from fl_manager.core.components.splitters.dataset_splitter_registry import (
    DatasetSplitterRegistry,
)
from fl_manager.core.utils.import_utils import ImportUtils

__all__ = [
    'DatasetSplitter',
    'DatasetSplitterRegistry',
    'BaseProportionDatasetSplitter',
]

ImportUtils.dynamic_registry_item_import('splitters', 'components')

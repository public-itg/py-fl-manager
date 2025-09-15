from fl_manager.core.components.readers.dataset_reader import DatasetReader
from fl_manager.core.components.readers.dataset_reader_registry import (
    DatasetReaderRegistry,
)
from fl_manager.core.utils.import_utils import ImportUtils

__all__ = ['DatasetReader', 'DatasetReaderRegistry']

ImportUtils.dynamic_registry_item_import('readers', 'components')

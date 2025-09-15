from fl_manager.core.components.datasets.dataframe_dataset import DataFrameDataset
from fl_manager.core.components.datasets.dataframe_dataset_registry import (
    DataFrameDatasetRegistry,
)
from fl_manager.core.utils.import_utils import ImportUtils

__all__ = ['DataFrameDatasetRegistry', 'DataFrameDataset']

ImportUtils.dynamic_registry_item_import('datasets', 'components')

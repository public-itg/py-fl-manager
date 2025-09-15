from fl_manager.core.components.datasets.dataframe_dataset import DataFrameDataset
from fl_manager.core.utils.registry import ClassRegistry

DataFrameDatasetRegistry = ClassRegistry[DataFrameDataset](DataFrameDataset)

from fl_manager.core.components.splitters.dataset_splitter import DatasetSplitter
from fl_manager.core.utils.registry import ClassRegistry

DatasetSplitterRegistry = ClassRegistry[DatasetSplitter](DatasetSplitter)

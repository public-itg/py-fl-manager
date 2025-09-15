from fl_manager.core.components.readers import DatasetReader
from fl_manager.core.utils.registry import ClassRegistry

DatasetReaderRegistry = ClassRegistry[DatasetReader](DatasetReader)

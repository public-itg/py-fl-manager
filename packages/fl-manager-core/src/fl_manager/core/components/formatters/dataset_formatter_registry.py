from fl_manager.core.components.formatters.dataset_formatter import DatasetFormatter
from fl_manager.core.utils.registry import ClassRegistry

DatasetFormatterRegistry = ClassRegistry[DatasetFormatter](DatasetFormatter)

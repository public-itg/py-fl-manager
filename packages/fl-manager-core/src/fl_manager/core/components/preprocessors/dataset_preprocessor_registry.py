from fl_manager.core.components.preprocessors.dataset_preprocessor import (
    DatasetPreprocessor,
)
from fl_manager.core.utils.registry import ClassRegistry

DatasetPreprocessorRegistry = ClassRegistry[DatasetPreprocessor](DatasetPreprocessor)

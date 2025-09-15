from fl_manager.core.components.preprocessors.dataset_preprocessor import (
    DatasetPreprocessor,
    DatasetPreprocessorComposite,
)
from fl_manager.core.components.preprocessors.dataset_preprocessor_registry import (
    DatasetPreprocessorRegistry,
)
from fl_manager.core.utils.import_utils import ImportUtils

__all__ = [
    'DatasetPreprocessor',
    'DatasetPreprocessorComposite',
    'DatasetPreprocessorRegistry',
]

ImportUtils.dynamic_registry_item_import('preprocessors', 'components')

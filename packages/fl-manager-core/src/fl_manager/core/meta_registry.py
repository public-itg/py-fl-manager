from fl_manager.core.components.checkers import DatasetCheckerRegistry
from fl_manager.core.components.datasets import DataFrameDatasetRegistry
from fl_manager.core.components.distributors import DatasetDistributorRegistry
from fl_manager.core.components.formatters import DatasetFormatterRegistry
from fl_manager.core.components.models import FederatedLearningModelRegistry
from fl_manager.core.components.preprocessors import DatasetPreprocessorRegistry
from fl_manager.core.components.readers import DatasetReaderRegistry
from fl_manager.core.components.splitters import DatasetSplitterRegistry
from fl_manager.core.components.validators import DatasetValidatorRegistry
from fl_manager.core.executors import FLExecutorRegistry
from fl_manager.core.runners import FLRunnerRegistry
from fl_manager.core.utils.registry import ClassRegistry, InstanceRegistry

# Initialize MetaRegistry
MetaRegistry = InstanceRegistry[ClassRegistry](ClassRegistry, allow_replacements=False)
# First Order Components
MetaRegistry.register(name='dataframe_dataset')(DataFrameDatasetRegistry)
MetaRegistry.register(name='dataset_distributor')(DatasetDistributorRegistry)
MetaRegistry.register(name='dataset_formatter')(DatasetFormatterRegistry)
MetaRegistry.register(name='dataset_reader')(DatasetReaderRegistry)
MetaRegistry.register(name='dataset_splitter')(DatasetSplitterRegistry)
MetaRegistry.register(name='dataset_validator')(DatasetValidatorRegistry)
MetaRegistry.register(name='fl_model')(FederatedLearningModelRegistry)
# Dependencies
MetaRegistry.register(name='dataset_checker')(DatasetCheckerRegistry)
MetaRegistry.register(name='dataset_preprocessor')(DatasetPreprocessorRegistry)
# Executor
MetaRegistry.register(name='fl_executor')(FLExecutorRegistry)
# Runner
MetaRegistry.register(name='fl_runner')(FLRunnerRegistry)

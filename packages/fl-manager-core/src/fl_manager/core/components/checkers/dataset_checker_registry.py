from fl_manager.core.components.checkers.dataset_checker import DatasetChecker
from fl_manager.core.utils.registry import ClassRegistry

DatasetCheckerRegistry = ClassRegistry[DatasetChecker](DatasetChecker)

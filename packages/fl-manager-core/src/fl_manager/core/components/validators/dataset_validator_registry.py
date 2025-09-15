from fl_manager.core.components.validators.dataset_validator import DatasetValidator
from fl_manager.core.utils.registry import ClassRegistry

DatasetValidatorRegistry = ClassRegistry[DatasetValidator](DatasetValidator)

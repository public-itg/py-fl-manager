from fl_manager.core.components.distributors.dataset_distributor import (
    DatasetDistributor,
)
from fl_manager.core.utils.registry import ClassRegistry

DatasetDistributorRegistry = ClassRegistry[DatasetDistributor](DatasetDistributor)

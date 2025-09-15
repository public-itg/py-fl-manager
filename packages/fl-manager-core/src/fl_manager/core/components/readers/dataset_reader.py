import abc
from typing import TYPE_CHECKING, Optional

from fl_manager.core.component import Component
from fl_manager.core.schemas.dataset import DatasetMapping

if TYPE_CHECKING:
    from fl_manager.core.schemas.pandas_dataset import PandasDataset


class DatasetReader(Component, metaclass=abc.ABCMeta):
    def __init__(self, dataset_mapping: Optional[dict | DatasetMapping] = None):
        if isinstance(dataset_mapping, dict):
            dataset_mapping = DatasetMapping(**dataset_mapping)
        self._dataset_mapping: DatasetMapping | None = dataset_mapping

    @property
    def dataset_mapping(self) -> DatasetMapping:
        if self._dataset_mapping is None:
            raise ValueError('no dataset split mapping defined')
        return self._dataset_mapping

    @abc.abstractmethod
    def fetch_dataset(self) -> 'PandasDataset':
        raise NotImplementedError()

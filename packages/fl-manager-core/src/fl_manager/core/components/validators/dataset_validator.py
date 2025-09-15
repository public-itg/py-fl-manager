import abc
from typing import TYPE_CHECKING

from fl_manager.core.component import Component

if TYPE_CHECKING:
    from fl_manager.core.schemas.pandas_dataset import PandasDataset


class DatasetValidator(Component, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def validate(self, in_data: 'PandasDataset'):
        raise NotImplementedError()

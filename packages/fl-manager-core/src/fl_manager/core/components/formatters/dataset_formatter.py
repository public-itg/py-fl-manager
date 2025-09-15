import abc
import logging
from typing import TYPE_CHECKING, Optional, List

from fl_manager.core.component import Component

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pandas import DataFrame


class DatasetFormatter(Component, metaclass=abc.ABCMeta):
    @property
    def is_composite(self) -> bool:
        return False

    @abc.abstractmethod
    def run(self, in_data: 'DataFrame') -> 'DataFrame':
        raise NotImplementedError()


class BaseDatasetFormatter(DatasetFormatter, metaclass=abc.ABCMeta):
    def __init__(self, in_col_name: str, out_col_name: Optional[str] = None):
        self._in_col_name = in_col_name
        self._out_col_name = out_col_name

    def run(self, in_data: 'DataFrame') -> 'DataFrame':
        """
        Args:
            in_data: Input DataFrame.

        Returns:
            DataFrame: After formatting.
        """
        if self._in_col_name == self._out_col_name:
            logger.info(f'Performing operation inplace on {self._in_col_name}.')
        return self._run_formatter(in_data)

    @abc.abstractmethod
    def _run_formatter(self, in_data: 'DataFrame') -> 'DataFrame':
        raise NotImplementedError()


class DatasetFormatterComposite(DatasetFormatter):
    def __init__(self):
        self._children: List[DatasetFormatter] = []

    @property
    def is_composite(self) -> bool:
        return True

    def add(self, dataset_formatter: DatasetFormatter):
        assert isinstance(dataset_formatter, DatasetFormatter), (
            f'invalid type {type(dataset_formatter)}'
        )
        self._children.append(dataset_formatter)

    def run(self, in_data: 'DataFrame') -> 'DataFrame':
        df = in_data
        for operation in self._children:
            df = operation.run(df)
        return df

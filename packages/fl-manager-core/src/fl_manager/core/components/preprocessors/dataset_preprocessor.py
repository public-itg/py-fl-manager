import abc
import logging
from typing import Any, List

from fl_manager.core.component import Component

logger = logging.getLogger(__name__)


class DatasetPreprocessor(Component, metaclass=abc.ABCMeta):
    @property
    def is_composite(self) -> bool:
        return False

    @abc.abstractmethod
    def preprocess(self, in_data: Any) -> Any:
        raise NotImplementedError()


class DatasetPreprocessorComposite(DatasetPreprocessor):
    def __init__(self):
        self._children: List[DatasetPreprocessor] = []

    @property
    def is_composite(self) -> bool:
        return True

    def add(self, dataset_preprocessor: DatasetPreprocessor):
        assert isinstance(dataset_preprocessor, DatasetPreprocessor), (
            f'invalid type {type(dataset_preprocessor)}'
        )
        self._children.append(dataset_preprocessor)

    def preprocess(self, in_data: Any) -> Any:
        data = in_data
        for preprocessor in self._children:
            data = preprocessor.preprocess(data)
        return data

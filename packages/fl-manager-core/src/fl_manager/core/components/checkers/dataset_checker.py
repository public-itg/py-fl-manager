import abc
from typing import TYPE_CHECKING

from fl_manager.core.component import Component

if TYPE_CHECKING:
    from pandera import Check


class DatasetChecker(Component, metaclass=abc.ABCMeta):
    """
    Contains logic for data checking to instantiate a `Check` object from pandera.
    """

    @abc.abstractmethod
    def get_checker(self) -> 'Check':
        raise NotImplementedError()

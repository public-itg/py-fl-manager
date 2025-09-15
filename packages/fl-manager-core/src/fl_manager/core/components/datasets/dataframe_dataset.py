import abc
import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Type, Callable

from fl_manager.core.component import Component
from fl_manager.core.schemas.dataset import GenericDataset

if TYPE_CHECKING:
    from fl_manager.core.schemas.pandas_dataset import PandasDataset

T = TypeVar('T')
logger = logging.getLogger(__name__)


class DataFrameDataset(Component, metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_cls: Type[T],
        dataset_kwargs: Optional[dict[str, Any]] = None,
        builder_func: Optional[Callable[..., T]] = None,
    ):
        assert inspect.isclass(dataset_cls), 'dataset_cls must be a class'
        self._dataset_cls = dataset_cls
        self._dataset_kwargs = dataset_kwargs or {}
        self._builder_func = builder_func

    @property
    @abc.abstractmethod
    def dataframe_kwarg_name(self) -> str:
        raise NotImplementedError()

    def get_dataset(self, dataset: 'PandasDataset') -> GenericDataset:
        return GenericDataset(
            train=self._get_dataset_instance(
                self._dataset_kwargs | {self.dataframe_kwarg_name: dataset.train}
            ),
            val=self._get_dataset_instance(
                self._dataset_kwargs | {self.dataframe_kwarg_name: dataset.val}
            )
            if dataset.val is not None and not dataset.val.empty
            else None,
            test=self._get_dataset_instance(
                self._dataset_kwargs | {self.dataframe_kwarg_name: dataset.test}
            )
            if dataset.test is not None and not dataset.test.empty
            else None,
        )

    def _get_dataset_instance(self, instance_kwargs: dict) -> T:
        _builder = (
            self._dataset_cls if self._builder_func is None else self._builder_func
        )
        return _builder(**instance_kwargs)

import abc
import inspect
from typing import List, Any, TypeVar, Type, Optional, Generic, Callable

from fl_manager.core.component import Component

T = TypeVar('T')


class FederatedLearningModel(Generic[T], Component, metaclass=abc.ABCMeta):
    def __init__(
        self,
        task: str,
        key_metric: str,
        negate_key_metric: bool,
        model_cls: Type[T],
        model_kwargs: Optional[dict[str, Any]] = None,
        builder_func: Optional[Callable[..., T]] = None,
        seed: Optional[int] = 42,
    ):
        assert inspect.isclass(model_cls), 'model_cls must be a class'
        assert task in self.spec_supported_tasks, (
            f'{model_cls.__name__} does not support task {task}.'
        )
        self._task = task
        self._key_metric = key_metric
        self._negate_key_metric = negate_key_metric
        self._model_cls = model_cls
        self._model_kwargs = model_kwargs or {}
        self._builder_func = builder_func
        self._seed = seed
        self._model = None

    @property
    def model_cls(self) -> Type[T]:
        return self._model_cls

    @property
    def key_metric(self) -> str:
        return self._key_metric

    @property
    def negate_key_metric(self) -> bool:
        return self._negate_key_metric

    @property
    @abc.abstractmethod
    def spec_supported_tasks(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def apply_seed(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_weights(self) -> dict:
        raise NotImplementedError()

    def get_model(self) -> T:
        if self._model is None:
            self.apply_seed()
            self._model = self._get_model_instance(self._model_kwargs)
        return self._model

    def _get_model_instance(self, instance_kwargs: dict) -> T:
        _builder = self._model_cls if self._builder_func is None else self._builder_func
        return _builder(**instance_kwargs)

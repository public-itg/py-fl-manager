import abc
import inspect
import logging
from typing import Type, TypeVar, List, Iterable, Generic, Callable, cast

T = TypeVar('T')
logger = logging.getLogger(__name__)


class BaseRegistry(Generic[T], metaclass=abc.ABCMeta):
    def __init__(
        self,
        registry_item_type: Type[T] | List[Type[T]],
        allow_replacements: bool = True,
    ):
        self._replaceable = allow_replacements
        self._is_multi_type = isinstance(registry_item_type, Iterable)
        self._registry_item_type = cast(
            List[Type[T]],
            (registry_item_type if self._is_multi_type else [registry_item_type]),
        )
        self._name = f'{self.__class__.__name__} for [{", ".join([ic.__name__ for ic in self._registry_item_type])}]'
        self._registry = {}

    def list(self) -> List[str]:
        return list(self._registry.keys())

    def _add_to_registry(
        self, item: T | Type[T], name: str, replace: bool = False
    ) -> None:
        if name in self._registry:
            if self._replaceable and replace:
                logger.warning(
                    f"Registry ({self._name}) value for '{name}' is being replaced!"
                )
            else:
                _msg = f"Registry ({self._name}) already has '{name}' registered."
                if not self._replaceable:
                    _msg = f'{_msg} And does not allow replacements!'
                raise RuntimeError(_msg)
        self._registry[name] = item

    @abc.abstractmethod
    def register(self, name: str, replace: bool = False) -> Callable:
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_registration_item(self, item: T | Type[T]):
        raise NotImplementedError()


class ClassRegistry(BaseRegistry[T]):
    def create(self, name: str, *args, **kwargs) -> T:
        _cls = self.get(name)
        return _cls(*args, **kwargs)

    def get(self, name: str) -> Type[T]:
        if (_item := self._registry.get(name)) is None:
            raise ValueError(f"'{name}' not found in registry ({self._name}).")
        return _item

    def register(self, name: str, replace: bool = False) -> Callable:
        def _decorator(item: Type[T]):
            self._validate_registration_item(item)
            self._add_to_registry(item, name, replace)
            return item

        return _decorator

    def _validate_registration_item(self, item: T | Type[T]):
        if not inspect.isclass(item):
            raise ValueError("Wrapped element must be a 'class'!")
        if not any([issubclass(item, ic) for ic in self._registry_item_type]):
            raise ValueError(
                f"Expecting '{self._name}' (or subclasses) but got '{item.__name__}'!"
            )


class InstanceRegistry(BaseRegistry[T]):
    def get(self, name: str) -> T:
        if (_item := self._registry.get(name)) is None:
            raise ValueError(f"'{name}' not found in registry ({self._name}).")
        return _item

    def register(self, name: str, replace: bool = False) -> Callable:
        def _decorator(item: T):
            self._validate_registration_item(item)
            self._add_to_registry(item, name, replace)
            return item

        return _decorator

    def _validate_registration_item(self, item: T | Type[T]):
        if not any([isinstance(item, rit) for rit in self._registry_item_type]):
            raise ValueError(
                f'Wrapped element must be a instance of {self._registry_item_type}!'
            )

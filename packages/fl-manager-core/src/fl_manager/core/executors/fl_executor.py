import abc
from typing import Any, Optional, TYPE_CHECKING, cast, List

from fl_manager.core.component import Component

if TYPE_CHECKING:
    from hamilton.driver import Driver


class FLExecutor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()


class BaseFLExecutor(FLExecutor, metaclass=abc.ABCMeta):
    def __init__(
        self,
        fl_train_id: str,
        components: dict[str, Component],
        fl_algorithm: str,
        fl_algorithm_kwargs: Optional[dict[str, Any]] = None,
    ):
        self._fl_train_id = fl_train_id
        self._components = components
        self._fl_algorithm = fl_algorithm
        self._fl_algorithm_kwargs = fl_algorithm_kwargs or {}

    @property
    @abc.abstractmethod
    def targets(self) -> list[str]:
        raise NotImplementedError()

    def run(self):
        _dr = self._setup_driver()
        _dr_inputs = self._setup_driver_inputs()
        _available_targets = [
            e.name for e in _dr.list_available_variables() if not e.is_external_input
        ]
        assert set(self.targets).issubset(set(_available_targets)), (
            f'ensure driver has {self.targets} targets'
        )
        _results = _dr.execute(cast(List[Any], self.targets), inputs=_dr_inputs)
        self._run_executor(_results)

    @abc.abstractmethod
    def _run_executor(self, data: dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def _setup_driver(self) -> 'Driver':
        raise NotImplementedError()

    @abc.abstractmethod
    def _setup_driver_inputs(self) -> dict:
        raise NotImplementedError()

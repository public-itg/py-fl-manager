from functools import cached_property
from typing import TYPE_CHECKING

from fl_manager.core.schemas.base_variables import BaseVariables
from fl_manager.core.schemas.registry_item import RegistryItem

if TYPE_CHECKING:
    from fl_manager.core.executors import FLExecutor


class FLConfig(BaseVariables):
    fl_executor: RegistryItem

    @cached_property
    def fl_executor_instance(self) -> 'FLExecutor':
        from fl_manager.core.executors import FLExecutor

        instance = self.fl_executor.instance
        assert isinstance(instance, FLExecutor)
        return instance

from importlib import import_module
from typing import TYPE_CHECKING, List, Any

from fl_manager.core.components.checkers import DatasetChecker, DatasetCheckerRegistry

if TYPE_CHECKING:
    from pandera import Check


@DatasetCheckerRegistry.register(name='is_instance')
class IsInstance(DatasetChecker):
    def __init__(self, instance_cls: List[str]):
        self._instance_cls = tuple(
            [self._get_instance_cls(cls) for cls in instance_cls]
        )

    def get_checker(self) -> 'Check':
        from pandera import Check

        return Check(
            check_fn=lambda s: all([isinstance(e, self._instance_cls) for e in s]),
            name=f'is_instance {self._instance_cls}',
        )

    @staticmethod
    def _get_instance_cls(instance_name: str) -> Any:
        _def = instance_name.rsplit('.', maxsplit=1)[::-1]
        cls_name = _def[0]
        pkg = _def[1] if len(_def) == 2 else 'builtins'
        _instance_cls = getattr(import_module(pkg), cls_name, None)
        if _instance_cls is None:
            raise ValueError(f'Could not find {cls_name} in (module) {pkg}.')
        return _instance_cls

import inspect
from typing import Any, Callable

import pytorch_lightning as pl
from lightning import Callback


def _forward_hook(name: str) -> Callable:
    def method(self, *args: Any, **kwargs: Any) -> Any:
        hook = getattr(self._callback, name, None)
        if hook is None:
            return getattr(self, name)(*args, **kwargs)
        if callable(hook):
            return hook(*args, **kwargs)

    return method


class LightningCallbackWrapper(Callback):
    def __init__(self, callback: pl.Callback):
        super().__init__()
        self._callback = callback

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        for name, _ in inspect.getmembers(Callback, predicate=inspect.isfunction):
            setattr(cls, name, _forward_hook(name))

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__'):
            raise AttributeError(name)
        if hasattr(self._callback, name):
            return getattr(self._callback, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __dir__(self) -> list[str]:
        return list(set(list(super().__dir__()) + dir(self.legacy_callback)))

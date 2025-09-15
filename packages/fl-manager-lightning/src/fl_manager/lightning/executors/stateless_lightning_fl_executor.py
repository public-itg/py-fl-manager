from typing import Type, TYPE_CHECKING

from fl_manager.core.executors import FLExecutorRegistry
from fl_manager.lightning.executors.lightning_fl_executor import LightningFLExecutor

if TYPE_CHECKING:
    from fl_manager.lightning.executors.nv.nvflare_lightning_executor import (
        NVFlareLightningExecutor,
    )


@FLExecutorRegistry.register(name='stateless_lightning_fl_executor')
class StatelessLightningFLExecutor(LightningFLExecutor):
    @property
    def nvflare_executor_cls(self) -> Type['NVFlareLightningExecutor']:
        from fl_manager.lightning.executors.nv.stateless_nvflare_lightning_executor import (
            StatelessNVFlareLightningExecutor,
        )

        return StatelessNVFlareLightningExecutor

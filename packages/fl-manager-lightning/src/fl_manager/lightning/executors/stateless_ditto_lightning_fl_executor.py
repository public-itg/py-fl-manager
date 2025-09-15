from typing import Type, TYPE_CHECKING, Optional, Any

from fl_manager.core.component import Component
from fl_manager.core.executors import FLExecutorRegistry
from fl_manager.lightning.executors.lightning_fl_executor import LightningFLExecutor

if TYPE_CHECKING:
    from fl_manager.lightning.executors.nv.stateless_ditto_nvflare_lightning_executor import (
        StatelessDittoNVFlareLightningExecutor,
    )


@FLExecutorRegistry.register(name='stateless_ditto_lightning_fl_executor')
class StatelessDittoLightningFLExecutor(LightningFLExecutor):
    def __init__(
        self,
        fl_train_id: str,
        components: dict[str, Component],
        best_ckpt_kwargs: dict,
        ditto_kwargs: dict,
        fl_algorithm: str,
        fl_algorithm_kwargs: Optional[dict[str, Any]] = None,
        trainer_kwargs: Optional[dict[str, Any]] = None,
        datamodule_name: Optional[str] = 'default',
        datamodule_kwargs: Optional[dict[str, Any]] = None,
        seed: Optional[int] = 42,
        deterministic: Optional[bool] = True,
    ):
        super().__init__(
            fl_train_id=fl_train_id,
            components=components,
            best_ckpt_kwargs=best_ckpt_kwargs,
            fl_algorithm=fl_algorithm,
            fl_algorithm_kwargs=fl_algorithm_kwargs,
            trainer_kwargs=trainer_kwargs,
            datamodule_name=datamodule_name,
            datamodule_kwargs=datamodule_kwargs,
            seed=seed,
            deterministic=deterministic,
        )
        self._ditto_kwargs = ditto_kwargs

    @property
    def nvflare_executor_cls(self) -> Type['StatelessDittoNVFlareLightningExecutor']:
        from fl_manager.lightning.executors.nv.stateless_ditto_nvflare_lightning_executor import (
            StatelessDittoNVFlareLightningExecutor,
        )

        return StatelessDittoNVFlareLightningExecutor

    def _run_executor(self, data: dict):
        self.nvflare_executor_cls(
            fl_train_id=self._fl_train_id,
            datamodule=data['datamodule'],
            model=data['model'],
            trainer_kwargs=self._trainer_kwargs,
            best_ckpt_kwargs=self._best_ckpt_kwargs,
            fl_algorithm=self._fl_algorithm,
            fl_algorithm_kwargs=self._fl_algorithm_kwargs,
            ditto_kwargs=self._ditto_kwargs,
            deterministic=self._deterministic,
            seed=self._seed,
        ).start()

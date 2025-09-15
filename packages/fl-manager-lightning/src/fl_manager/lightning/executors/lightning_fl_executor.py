import abc
from typing import TYPE_CHECKING, Any, Optional, List, Type

from fl_manager.core.component import Component
from fl_manager.core.executors.fl_executor import BaseFLExecutor

if TYPE_CHECKING:
    from hamilton.driver import Driver
    from fl_manager.lightning.executors.nv.nvflare_lightning_executor import (
        NVFlareLightningExecutor,
    )


class LightningFLExecutor(BaseFLExecutor, metaclass=abc.ABCMeta):
    def __init__(
        self,
        fl_train_id: str,
        components: dict[str, Component],
        best_ckpt_kwargs: dict,
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
            fl_algorithm=fl_algorithm,
            fl_algorithm_kwargs=fl_algorithm_kwargs,
        )
        self._trainer_kwargs = trainer_kwargs or {}
        self._best_ckpt_kwargs = best_ckpt_kwargs
        self._datamodule_name = datamodule_name
        self._datamodule_kwargs = datamodule_kwargs or {}
        self._seed = seed
        self._deterministic = deterministic

    @property
    def targets(self) -> List[str]:
        return ['datamodule', 'model']

    @property
    @abc.abstractmethod
    def nvflare_executor_cls(self) -> Type['NVFlareLightningExecutor']:
        raise NotImplementedError()

    def _run_executor(self, data: dict):
        self.nvflare_executor_cls(
            fl_train_id=self._fl_train_id,
            datamodule=data['datamodule'],
            model=data['model'],
            trainer_kwargs=self._trainer_kwargs,
            best_ckpt_kwargs=self._best_ckpt_kwargs,
            fl_algorithm=self._fl_algorithm,
            fl_algorithm_kwargs=self._fl_algorithm_kwargs,
            deterministic=self._deterministic,
            seed=self._seed,
        ).start()

    def _setup_driver(self) -> 'Driver':
        from hamilton import driver
        from fl_manager.core.dataflows import dataset_setup
        from fl_manager.lightning.dataflows import base_lightning_trainer

        return (
            driver.Builder()
            .with_modules(dataset_setup, base_lightning_trainer)
            .with_config({})  # replace with configuration as appropriate
            .build()
        )

    def _setup_driver_inputs(self) -> dict:
        return {
            'fl_train_id': self._fl_train_id,
            'fl_algorithm': self._fl_algorithm,
            'fl_algorithm_kwargs': self._fl_algorithm_kwargs,
            'trainer_kwargs': self._trainer_kwargs,
            'datamodule_name': self._datamodule_name,
            'datamodule_kwargs': self._datamodule_kwargs,
            'deterministic': self._deterministic,
            'seed': self._seed,
            **self._components,
        }

import logging
from pathlib import Path
from typing import Optional, Literal

import nvflare.client.lightning as flare
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel

from fl_manager.core.constants import FL_MANAGER_HOME
from fl_manager.lightning.components.callbacks import LightningCallbackRegistry
from fl_manager.lightning.executors.nv.stateless_nvflare_lightning_executor import (
    StatelessNVFlareLightningExecutor,
)

logger = logging.getLogger(__name__)


class StatelessDittoNVFlareLightningExecutor(StatelessNVFlareLightningExecutor):
    """
    Ditto Executor

    - Li et al. 2021 - [`Ditto`: Fair and Robust Federated Learning Through Personalization](https://arxiv.org/pdf/2012.04221)

    Each round an only local model is optimized with FedProx term w.r.t. global model.

    Based on nvflare [DittoHelper](https://nvflare.readthedocs.io/en/2.5.2/_modules/nvflare/app_opt/pt/ditto.html).
    """

    class DittoParams(BaseModel):
        ditto_lambda: float
        epochs: int
        mode: Literal['max', 'min']
        monitor: str

    def __init__(
        self,
        fl_train_id: str,
        datamodule: LightningDataModule,
        model: LightningModule,
        trainer_kwargs: dict,
        best_ckpt_kwargs: dict,
        fl_algorithm: str,
        fl_algorithm_kwargs: dict,
        ditto_kwargs: dict,
        seed: int,
        deterministic: Optional[bool] = True,
    ):
        super().__init__(
            fl_train_id=fl_train_id,
            datamodule=datamodule,
            model=model,
            trainer_kwargs=trainer_kwargs,
            best_ckpt_kwargs=best_ckpt_kwargs,
            fl_algorithm=fl_algorithm,
            fl_algorithm_kwargs=fl_algorithm_kwargs,
            seed=seed,
            deterministic=deterministic,
        )
        self._ditto_kwargs = self.DittoParams.model_validate(ditto_kwargs)

    @property
    def ditto_run_dir(self) -> Path:
        return self.run_base_dir / 'ditto'

    def _train_task_handler(self):
        input_model = flare.receive()
        if input_model is None:
            return
        self._train_ditto_model(
            f'ditto_{self._version_prefix}_{input_model.current_round}'
        )
        super()._train_task_handler()

    def _export_model_task_handler(self):
        self._train_ditto_model('ditto_export_0')
        super()._export_model_task_handler()

    def _fit_and_export_model_task_handler(self):
        self._train_ditto_model('ditto_fit_export_0')
        super()._fit_and_export_model_task_handler()

    def _train_ditto_model(self, version):
        ditto_trainer = Trainer(
            deterministic=self._deterministic,
            max_epochs=self._ditto_kwargs.epochs,
            logger=TensorBoardLogger(
                save_dir=FL_MANAGER_HOME / 'logs',
                name=f'{self._fl_train_id}__{self._job_id}',
                version=version,
            ),
            callbacks=[
                LightningCallbackRegistry.create(
                    'ditto',
                    ditto_lambda=self._ditto_kwargs.ditto_lambda,
                    save_dir=self.ditto_run_dir,
                    monitor=self._ditto_kwargs.monitor,
                    mode=self._ditto_kwargs.mode,
                )
            ],
        )
        logger.info('--- train ditto model ---')
        ditto_trainer.fit(self._model, datamodule=self._datamodule)

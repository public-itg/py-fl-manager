import logging
from copy import deepcopy
from typing import Dict

import nvflare.client.lightning as flare
from lightning import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client import send

from fl_manager.core.constants import FL_MANAGER_HOME
from fl_manager.lightning.components.callbacks.nvflare_callbacks_wrappers import (
    WrappedRestoreStateCallback,
)
from fl_manager.lightning.executors.nv.stateless_nvflare_lightning_executor import (
    StatelessNVFlareLightningExecutor,
)

logger = logging.getLogger(__name__)


class StatefulNVFlareLightningExecutor(StatelessNVFlareLightningExecutor):
    """
    Stateful Executor
    - Use same trainer during all training process.
    """

    @property
    def _version_prefix(self) -> str:
        return 'version'

    def _init_callbacks(self) -> Dict[str, Callback]:
        restore_state_callback = WrappedRestoreStateCallback()
        model_checkpoint_callback = ModelCheckpoint(save_last='link')
        best_model_checkpoint_callback = ModelCheckpoint(
            **(
                self._best_ckpt_kwargs
                | {'dirpath': self.run_base_dir / 'checkpoints', 'save_last': False}
            )
        )
        return {
            'fl_callback': self._fl_callback,
            'restore_state': restore_state_callback,
            'model_checkpoint': model_checkpoint_callback,
            'best_model_checkpoint': best_model_checkpoint_callback,
        }

    def _start_nvflare_executor(self):
        self._trainer = Trainer(
            deterministic=self._deterministic,
            **(
                self._trainer_kwargs
                | {
                    'logger': TensorBoardLogger(
                        save_dir=FL_MANAGER_HOME / 'logs',
                        name=f'{self._fl_train_id}__{self._job_id}',
                    ),
                    'callbacks': deepcopy(list(self._callbacks.values())),
                }
            ),
        )
        self._run_nvflare_loop()

    def _train_task_handler(self):
        input_model = flare.receive()
        if input_model is None:
            return
        logger.info(
            f'[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]'
        )
        logger.info('--- validate global model ---')
        self._trainer.validate(self._model, datamodule=self._datamodule)
        logger.info('--- train new model ---')
        self._trainer.fit(self._model, datamodule=self._datamodule)

    def _fit_and_export_model_task_handler(self):
        self._trainer.validate(self._model, datamodule=self._datamodule)
        self._trainer.fit(self._model, datamodule=self._datamodule)
        self._export_pfl_best_model()
        send(FLModel(metrics={}))

import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict

import nvflare.client.lightning as flare
import torch
from lightning import Trainer, seed_everything, Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.api import send

from fl_manager.core.constants import FL_MANAGER_HOME
from fl_manager.lightning.components.callbacks import LightningCallbackRegistry
from fl_manager.lightning.executors.nv.nvflare_lightning_executor import (
    NVFlareLightningExecutor,
)

logger = logging.getLogger(__name__)


class StatelessNVFlareLightningExecutor(NVFlareLightningExecutor):
    """
    Stateless Executor
    - Each round starts with a new Trainer (clean states every round).
    """

    @property
    def _version_prefix(self) -> str:
        return 'round'

    def start(self):
        self._fl_callback = self._get_fl_callback()
        # client_api gets initialized inside FLCallback initialization
        self._job_id = flare.get_job_id()
        self._callbacks = self._init_callbacks()
        seed_everything(self._seed, workers=True)
        self._start_nvflare_executor()

    def _get_fl_callback(self) -> Callback:
        _default_trainer = Trainer()
        _fl_callback_kwargs = self._fl_algorithm_kwargs | {
            'rank': _default_trainer.global_rank,
            'load_state_dict_strict': False,
        }
        return LightningCallbackRegistry.get(self._fl_algorithm)(**_fl_callback_kwargs)

    def _init_callbacks(self) -> Dict[str, Callback]:
        model_checkpoint_callback = ModelCheckpoint(save_last='link')
        # TODO: validate kwargs have the expected input
        best_model_checkpoint_callback = ModelCheckpoint(
            **(
                self._best_ckpt_kwargs
                | {'dirpath': self.run_base_dir / 'checkpoints', 'save_last': False}
            )
        )
        return {
            'fl_callback': self._fl_callback,
            'model_checkpoint': model_checkpoint_callback,
            'best_model_checkpoint': best_model_checkpoint_callback,
        }

    def _start_nvflare_executor(self):
        self._run_nvflare_loop()

    def _train_task_handler(self):
        input_model = flare.receive()
        if input_model is None:
            return
        logger.info(
            f'[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]'
        )
        trainer = Trainer(
            deterministic=self._deterministic,
            **(
                self._trainer_kwargs
                | {
                    'logger': TensorBoardLogger(
                        save_dir=FL_MANAGER_HOME / 'logs',
                        name=f'{self._fl_train_id}__{self._job_id}',
                        version=f'{self._version_prefix}_{input_model.current_round}',
                    ),
                    'callbacks': deepcopy(list(self._callbacks.values())),
                }
            ),
        )
        logger.info('--- validate global model ---')
        trainer.validate(self._model, datamodule=self._datamodule)
        logger.info('--- train new model ---')
        trainer.fit(self._model, datamodule=self._datamodule)

    def _validation_task_handler(self):
        input_model = flare.receive()
        if input_model is None:
            return
        logger.info(f'Validating {input_model.meta.get("model_name")}')
        trainer = Trainer(
            deterministic=self._deterministic,
            logger=False,
            callbacks=deepcopy(self._callbacks.get('fl_callback')),
        )
        trainer.validate(
            self._model, datamodule=self._datamodule, ckpt_path=self._get_ckpt_path()
        )

    def _submit_model_task_handler(self):
        params = torch.load(
            self._get_ckpt_path(), map_location='cpu', weights_only=True
        ).get('state_dict')
        send(FLModel(params=params, meta={}))

    def _export_model_task_handler(self):
        trainer = Trainer(
            deterministic=self._deterministic,
            logger=False,
            enable_model_summary=False,
            callbacks=deepcopy(self._callbacks.get('fl_callback')),
        )
        trainer.validate(
            self._model,
            datamodule=self._datamodule,
            ckpt_path=self._get_ckpt_path(),
            verbose=False,
        )
        trainer.save_checkpoint(
            self.run_base_dir / 'results/model.ckpt', weights_only=True
        )
        send(FLModel(metrics={}))

    def _fit_and_export_model_task_handler(self):
        trainer = Trainer(
            deterministic=self._deterministic,
            enable_model_summary=False,
            **(
                self._trainer_kwargs
                | {
                    'logger': TensorBoardLogger(
                        save_dir=FL_MANAGER_HOME / 'logs',
                        name=f'{self._fl_train_id}__{self._job_id}',
                        version='fit_export_0',
                    ),
                    'callbacks': deepcopy(list(self._callbacks.values())),
                }
            ),
        )
        trainer.validate(self._model, datamodule=self._datamodule)
        trainer.fit(self._model, datamodule=self._datamodule)
        self._export_pfl_best_model()
        send(FLModel(metrics={}))

    def _export_pfl_best_model(self):
        _best_checkpoints = Path(self.run_base_dir / 'checkpoints').glob('*.ckpt')
        data = {
            v.get('kth_best_model_path'): v.get('kth_value').item()
            for e in _best_checkpoints
            for v in torch.load(e, map_location='cpu', weights_only=True)
            .get('callbacks')
            .values()
            if v.get('kth_best_model_path')
        }
        descending = self._best_ckpt_kwargs.get('mode') == 'max'
        _best_ckpt = sorted(data, key=data.get, reverse=descending)[0]
        _target = self.run_base_dir / 'results'
        _target.mkdir(parents=True, exist_ok=True)
        shutil.copy(_best_ckpt, _target / 'pfl_model.ckpt')

    def _get_ckpt_path(self) -> Path:
        return (
            self.run_base_dir
            / f'{self._version_prefix}_{self._get_last_version()}/checkpoints/last.ckpt'
        )

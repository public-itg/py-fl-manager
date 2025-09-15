import logging
import operator
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
from nvflare.app_common.abstract.fl_model import FLModel

from fl_manager.lightning.components.callbacks import LightningCallbackRegistry
from fl_manager.lightning.components.callbacks.base.fed_prox_fl_callback import (
    _FedProxFLCallback,
)
from fl_manager.lightning.utils.lightning_callback_wrapper import (
    LightningCallbackWrapper,
)
from fl_manager.lightning.utils.lightning_utils import LightningUtils

logger = logging.getLogger(__name__)


class _DittoFLCallback(_FedProxFLCallback):
    def __init__(
        self,
        ditto_lambda: float,
        save_dir: str,
        monitor: str,
        mode: Literal['max', 'min'],
        rank: int = 0,
        load_state_dict_strict: bool = True,
    ):
        assert mode in ['max', 'min'], f'invalid mode {mode}'
        super().__init__(
            mu=ditto_lambda, rank=rank, load_state_dict_strict=load_state_dict_strict
        )
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._monitor = monitor
        self._mode = mode
        self._best_metric: int = 0
        self._operator = operator.gt if mode == 'max' else operator.lt
        self._model_file_path = self._save_dir / 'ditto_model.pt'
        self._best_model_file_path = self._save_dir / 'ditto_best_model.pt'

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super().on_train_start(trainer, pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        torch.save(
            {
                'state_dict': {
                    k: v.cpu() for k, v in trainer.model.state_dict().items()
                },
                'metric': LightningUtils.get_metrics(trainer.callback_metrics).get(
                    self._monitor
                ),
            },
            f=self._model_file_path,
        )
        super().on_train_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if pl_module:
            self.metrics = LightningUtils.get_metrics(trainer.callback_metrics)
            _metric = self.metrics.get(self._monitor)
            if self._operator(_metric, self._best_metric):
                logger.info(
                    f'New best metric (for {self._monitor}): {_metric} (old: {self._best_metric})'
                )
                self._best_metric = _metric
                torch.save(
                    {
                        'state_dict': {
                            k: v.cpu() for k, v in trainer.model.state_dict().items()
                        },
                        'metric': self._best_metric,
                    },
                    f=self._best_model_file_path,
                )

    def _receive_and_update_model(self, trainer, pl_module):
        model = self._receive_model(trainer)
        if model.current_round is not None:
            self.current_round = model.current_round
        if self._model_file_path.exists():
            params = (
                torch.load(self._model_file_path, map_location='cpu', weights_only=True)
            ).get('state_dict')
            pl_module.load_state_dict(params, strict=self._load_state_dict_strict)
        elif model.params:
            logger.info('Not local model yet. Loading from server.')
            pl_module.load_state_dict(model.params, strict=self._load_state_dict_strict)
        if self._best_model_file_path.exists():
            metric = (
                torch.load(
                    self._best_model_file_path, map_location='cpu', weights_only=True
                )
            ).get('metric')
            self._best_metric = metric

    def reset_state(self, trainer):
        """Local train only, avoid reset state."""
        pass

    def _send_model(self, output_model: FLModel):
        """Local train only, avoid sending anything."""
        pass


@LightningCallbackRegistry.register(name='ditto')
class DittoFLCallback(LightningCallbackWrapper):
    def __init__(
        self,
        ditto_lambda: float,
        save_dir: str,
        monitor: str,
        mode: Literal['max', 'min'],
        rank: int = 0,
        load_state_dict_strict: bool = True,
    ):
        super().__init__(
            callback=_DittoFLCallback(
                ditto_lambda=ditto_lambda,
                save_dir=save_dir,
                monitor=monitor,
                mode=mode,
                rank=rank,
                load_state_dict_strict=load_state_dict_strict,
            )
        )

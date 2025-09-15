import abc
import logging
from pathlib import Path
from typing import Optional, Dict

import nvflare.client.lightning as flare
from lightning import (
    LightningDataModule,
    LightningModule,
    seed_everything,
    Callback,
    Trainer,
)

from fl_manager.core.constants import FL_MANAGER_HOME
from fl_manager.core.executors.nv.nvflare_client_api_executor import (
    BaseNVFlareClientAPIExecutor,
)
from fl_manager.lightning.components.callbacks import LightningCallbackRegistry

logger = logging.getLogger(__name__)


class NVFlareLightningExecutor(BaseNVFlareClientAPIExecutor, metaclass=abc.ABCMeta):
    def __init__(
        self,
        fl_train_id: str,
        datamodule: LightningDataModule,
        model: LightningModule,
        trainer_kwargs: dict,
        best_ckpt_kwargs: dict,
        fl_algorithm: str,
        fl_algorithm_kwargs: dict,
        seed: int,
        deterministic: Optional[bool] = True,
    ):
        super().__init__()
        self._fl_train_id = fl_train_id
        self._datamodule = datamodule
        self._model = model
        self._trainer_kwargs = trainer_kwargs
        self._best_ckpt_kwargs = best_ckpt_kwargs
        self._fl_algorithm = fl_algorithm
        self._fl_algorithm_kwargs = fl_algorithm_kwargs
        self._seed = seed
        self._deterministic = deterministic
        self._fl_callback = None
        self._job_id = None
        self._callbacks = None

    @property
    def run_base_dir(self) -> Path:
        if self._job_id is None:
            raise RuntimeError('Not started.')
        return FL_MANAGER_HOME / 'logs' / f'{self._fl_train_id}__{self._job_id}'

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

    @property
    @abc.abstractmethod
    def _version_prefix(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def _init_callbacks(self) -> Dict[str, Callback]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _start_nvflare_executor(self):
        raise NotImplementedError()

    def _get_last_version(self) -> int:
        return sorted(
            [
                int(str(e).rsplit('_', 1)[-1])
                for e in self.run_base_dir.rglob(f'{self._version_prefix}_*')
            ]
        )[-1]

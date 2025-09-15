import abc
import logging
from typing import Callable

from nvflare.app_common.app_constant import AppConstants
from nvflare.client.api import is_running, get_task_name

logger = logging.getLogger(__name__)


class NVFlareClientAPIExecutor(metaclass=abc.ABCMeta):
    def __init__(self, task_mapping: dict[str, Callable]):
        assert isinstance(task_mapping, dict), (
            f'expected dict but got {type(task_mapping)}'
        )
        assert all([callable(v) for v in task_mapping.values()])
        self._task_mapping = task_mapping

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError()

    def _run_nvflare_loop(self):
        while is_running():
            task_name = get_task_name()
            logger.info(f'=== TASK: {task_name} ===')
            _handler = self._task_mapping.get(task_name)
            if _handler is None:
                raise RuntimeError(f'Unknown task: {task_name}')
            _handler()


class BaseNVFlareClientAPIExecutor(NVFlareClientAPIExecutor, metaclass=abc.ABCMeta):
    """
    Base NVFlare Client API Executor.

    User must implement handlers for:
        - `AppConstants.TASK_TRAIN` (from nvflare)
        - `AppConstants.TASK_VALIDATION` (from nvflare)
        - `AppConstants.TASK_SUBMIT_MODEL` (from nvflare)
        - `'export_model'` (custom task)
        - `'fit_and_export_model'` (custom task)
    """

    def __init__(self):
        super().__init__(
            task_mapping={
                AppConstants.TASK_TRAIN: self._train_task_handler,
                AppConstants.TASK_VALIDATION: self._validation_task_handler,
                AppConstants.TASK_SUBMIT_MODEL: self._submit_model_task_handler,
                'export_model': self._export_model_task_handler,
                'fit_and_export_model': self._fit_and_export_model_task_handler,
            }
        )

    @abc.abstractmethod
    def _train_task_handler(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _validation_task_handler(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _submit_model_task_handler(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _export_model_task_handler(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _fit_and_export_model_task_handler(self):
        raise NotImplementedError()

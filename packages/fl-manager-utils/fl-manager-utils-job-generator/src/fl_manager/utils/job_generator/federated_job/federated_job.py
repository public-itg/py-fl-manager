import abc
from typing import Any, List

from nvflare import FedJob
from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.job_config.script_runner import ScriptRunner

from fl_manager.core.components.models import FederatedLearningModel


class FederatedJob(metaclass=abc.ABCMeta):
    def __init__(self, fl_model: FederatedLearningModel) -> None:
        self._fl_model = fl_model

    @property
    def initial_model(self) -> Any:
        return self._fl_model.get_model()

    @property
    def key_metric(self) -> str:
        return self._fl_model.key_metric

    @property
    def negate_key_metric(self) -> bool:
        return self._fl_model.negate_key_metric

    @abc.abstractmethod
    def get_fed_job(
        self, name: str, num_rounds: int, clients: List[str], fl_algorithm_kwargs: dict
    ) -> FedJob:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_script_runner(self, script: str, script_args: List[str]) -> ScriptRunner:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_params_converter(self, supported_tasks: list[str]) -> ParamsConverter:
        raise NotImplementedError()

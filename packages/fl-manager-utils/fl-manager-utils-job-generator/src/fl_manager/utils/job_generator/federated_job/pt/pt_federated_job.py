import abc
from typing import List

from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.app_opt.pt.params_converter import NumpyToPTParamsConverter
from nvflare.job_config.script_runner import FrameworkType

from fl_manager.utils.job_generator.federated_job.federated_job import FederatedJob
from fl_manager.utils.nvflare_extensions.pt.executors.pt_weight_initializer_executor import (
    PTWeightInitializerExecutor,
)


class PTFederatedJob(FederatedJob, metaclass=abc.ABCMeta):
    _FRAMEWORK = FrameworkType.PYTORCH

    def get_fed_job(
        self, name: str, num_rounds: int, clients: List[str], fl_algorithm_kwargs: dict
    ) -> BaseFedJob:
        _job = self._get_fed_job(name, num_rounds, clients, fl_algorithm_kwargs)
        self._add_export_controller(_job)
        _weight_initializer_executor = self._get_weight_initializer_executor()
        _from_params_converter = self._get_params_converter()
        for cl in clients:
            _job.to(
                obj=_weight_initializer_executor,
                target=cl,
                tasks=[AppConstants.TASK_GET_WEIGHTS],
            )
            _job.to(obj=_from_params_converter, target=cl, id='from_nvflare')
        return _job

    def get_params_converter(self, supported_tasks: list[str]) -> ParamsConverter:
        return NumpyToPTParamsConverter(supported_tasks)

    @abc.abstractmethod
    def _get_fed_job(
        self, name: str, num_rounds: int, clients: List[str], fl_algorithm_kwargs: dict
    ) -> BaseFedJob:
        raise NotImplementedError()

    @abc.abstractmethod
    def _add_export_controller(self, job: BaseFedJob):
        raise NotImplementedError()

    def _get_weight_initializer_executor(self):
        return PTWeightInitializerExecutor(fl_model=self._fl_model)

    def _get_params_converter(self):
        return self.get_params_converter(
            [
                AppConstants.TASK_TRAIN,
                AppConstants.TASK_VALIDATION,
                'export_model',
                'fit_and_export_model',
            ]
        )

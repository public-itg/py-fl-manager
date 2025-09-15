import logging
from typing import List, cast

from nvflare.app_common.workflows.initialize_global_weights import (
    InitializeGlobalWeights,
)
from nvflare.app_opt.pt import PTFileModelPersistor
from nvflare.app_opt.pt.in_process_client_api_executor import (
    PTInProcessClientAPIExecutor,
)
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, BaseScriptRunner

from fl_manager.utils.job_generator.federated_job.pt import PTFederatedJobRegistry
from fl_manager.utils.job_generator.federated_job.pt.fed_job.fed_opt import FedOptJob
from fl_manager.utils.job_generator.federated_job.pt.pt_federated_job import (
    PTFederatedJob,
)
from fl_manager.utils.nvflare_extensions.nv.workflows.model_export_controller import (
    ModelExportController,
)

logger = logging.getLogger(__name__)


@PTFederatedJobRegistry.register(name='fed_opt')
class PTFedOptJob(PTFederatedJob):
    def _get_fed_job(
        self, name: str, num_rounds: int, clients: List[str], fl_algorithm_kwargs: dict
    ) -> BaseFedJob:
        _optimizer_args = fl_algorithm_kwargs.pop('optimizer_args', None)
        if _optimizer_args is None:
            # Set default as in nvflare shareable, set here because of an error with the self.logger it calls there
            _optimizer_args = {
                'path': 'torch.optim.SGD',
                'args': {'lr': 1.0, 'momentum': 0.6},
                'config_type': 'dict',
            }
        _lr_scheduler_args = fl_algorithm_kwargs.pop('lr_scheduler_args', None)
        if len(fl_algorithm_kwargs) > 0:
            logger.warning(
                f'Unused kwargs at FedJob ({self.__class__.__name__}): {fl_algorithm_kwargs}'
            )
        _job = FedOptJob(
            initial_model=self.initial_model,
            n_clients=len(clients),
            num_rounds=num_rounds,
            name=name,
            min_clients=len(clients),
            mandatory_clients=clients,
            key_metric=self.key_metric,
            negate_key_metric=self.negate_key_metric,
            model_persistor=PTFileModelPersistor(),
            optimizer_args=_optimizer_args,
            lr_scheduler_args=_lr_scheduler_args,
            pre_train_controllers={'weights_initializer': InitializeGlobalWeights()},
        )
        return _job

    def get_script_runner(
        self, script: str, script_args: List[str]
    ) -> BaseScriptRunner:
        _executor = PTInProcessClientAPIExecutor(
            task_script_path=script,
            task_script_args=' '.join(script_args),
            params_exchange_format=ExchangeFormat.PYTORCH,
            params_transfer_type=TransferType.DIFF,
            from_nvflare_converter_id='from_nvflare',
        )
        return BaseScriptRunner(
            script=script,
            script_args=' '.join(script_args),
            framework=cast(FrameworkType, FrameworkType.PYTORCH),
            params_transfer_type=TransferType.DIFF,
            executor=_executor,
        )

    def _add_export_controller(self, job: BaseFedJob):
        job.to_server(
            ModelExportController(
                model_locator_id=job.comp_ids['locator_id'],
                key_metric=self.key_metric,
                negate_key_metric=self.negate_key_metric,
            ),
            'model_export_controller',
        )

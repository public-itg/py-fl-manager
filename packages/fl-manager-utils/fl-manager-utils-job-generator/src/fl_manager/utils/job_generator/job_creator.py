import json
import logging
import shutil
import tempfile
from pathlib import Path

from nvflare import FedJob
from nvflare.app_common.app_constant import AppConstants

from fl_manager.core.components.models import FederatedLearningModel
from fl_manager.core.runners import FLRunner, FLRunnerRegistry
from fl_manager.core.schemas.registry_item import RegistryItem
from fl_manager.utils.job_generator.federated_job.federated_job_factory import (
    FederatedJobFactory,
)
from fl_manager.utils.job_generator.schemas.client_config import ClientConfig
from fl_manager.utils.job_generator.schemas.job_config import JobConfig

logger = logging.getLogger(__name__)


class JobCreator:
    def __init__(self, job_config: JobConfig):
        assert isinstance(job_config, JobConfig), 'invalid job config'
        self._jc = job_config
        self._runner: type[FLRunner] = FLRunnerRegistry.get(self._jc.runner)
        self._job: FedJob | None = None
        self._runner_file: Path | None = None
        self._config_files: dict[str, Path] | None = None

    def create(self, output_path: str) -> Path:
        _output_path = Path(output_path)
        try:
            _job = self._create_job()
            _job.export_job(str(_output_path.resolve()))
            # Remove 'custom' related to fl_manager (if any)
            [shutil.rmtree(e) for e in _output_path.glob(r'**/fl_manager')]
        finally:
            self._clean_artifacts()
        return _output_path

    def simulator_run(
        self, simulator_dir: str, simulator_kwargs: dict | None = None
    ) -> Path:
        _simulation_path = Path(simulator_dir)
        _simulator_kwargs = (simulator_kwargs or {}) | {'workspace': simulator_dir}
        if _simulation_path.exists():
            logger.info(f'Deleting previous simulation at {simulator_dir}')
            shutil.rmtree(simulator_dir)
        try:
            _job = self._create_job()
            _job.simulator_run(**_simulator_kwargs)
        finally:
            self._clean_artifacts()
        return _simulation_path

    def _clean_artifacts(self):
        if self._runner_file is not None:
            self._runner_file.unlink(missing_ok=True)
        if self._config_files is not None:
            [
                _config_file.unlink(missing_ok=True)
                for _config_file in self._config_files.values()
            ]

    def _generate_artifacts(self):
        self._clean_artifacts()  # Ensure clean state
        self._runner_file: Path = self._get_runner_file()
        self._config_files: dict[str, Path] = {
            cl_name: self._get_config_file(cl_config)
            for cl_name, cl_config in self._jc.clients.items()
        }

    def _create_job(self) -> FedJob:
        self._generate_artifacts()
        _fl_model = self.get_model_from_config()
        federated_job = FederatedJobFactory.create(
            _fl_model, self._jc.fl_algorithm.name
        )
        _clients = self._jc.clients
        job = federated_job.get_fed_job(
            name=self._jc.name,
            num_rounds=self._jc.num_rounds,
            clients=list(_clients.keys()),
            fl_algorithm_kwargs=(self._jc.fl_algorithm.server_kwargs or {}),
        )
        for cl_name, cl_config in self._config_files.items():
            job.to(
                obj=federated_job.get_script_runner(
                    script=self._runner_file.name,
                    script_args=self._runner.get_script_args(cl_config.name),
                ),
                target=cl_name,
                tasks=[
                    AppConstants.TASK_TRAIN,
                    AppConstants.TASK_VALIDATION,
                    AppConstants.TASK_SUBMIT_MODEL,
                    'export_model',
                    'fit_and_export_model',
                ],
            )
            job.to(obj=cl_config.name, target=cl_name)
        return job

    def get_model_from_config(self) -> FederatedLearningModel:
        _fl_model = self._jc.components.get('fl_model')
        assert _fl_model is not None, 'Model must be specified in components.'
        assert isinstance(_fl_model, RegistryItem), (
            'invalid model config, cannot be list.'
        )
        return _fl_model.instance

    def _get_config_file(self, client_config: ClientConfig) -> Path:
        _components = self._jc.components | (client_config.components or {})
        data = {
            'fl_executor': RegistryItem(
                registry_id='fl_executor',
                name=self._jc.executor,
                keyword_arguments={
                    'components': _components,
                    'fl_algorithm': self._jc.fl_algorithm.name,
                    'fl_algorithm_kwargs': self._jc.fl_algorithm.client_kwargs,
                    **(self._jc.executor_kwargs or {}) | {'fl_train_id': self._jc.name},
                },
            ).model_dump()
        }
        with tempfile.NamedTemporaryFile(
            mode='w+', dir='./', suffix='.json', delete=False
        ) as temp_file:
            json.dump(data, temp_file, indent=4)
            return Path(temp_file.name)

    def _get_runner_file(self) -> Path:
        with tempfile.NamedTemporaryFile(
            mode='w+', dir='./', suffix='.py', delete=False
        ) as temp_file:
            temp_file.write(self._runner.to_script())
            return Path(temp_file.name)

import abc
import logging
import os
from pathlib import Path

from fl_manager.core.executors import FLExecutor

logger = logging.getLogger(__name__)


class FLRunner(metaclass=abc.ABCMeta):
    @classmethod
    def run(cls, workspace: str, site: str, job_id: str, config_path: str):
        _app_dir = Path(workspace) / job_id / f'app_{site}'
        config_full_path = _app_dir / config_path
        logger.info(f'Running {os.getcwd()}')
        logger.info(f'App Dir: {_app_dir}')
        assert config_full_path.exists(), f'{config_full_path} does not exist.'
        logger.info(f'Running config {config_full_path}...')
        _fl_executor = cls.get_fl_executor_from_config(
            config=cls.load_config(config_full_path)
        )
        _fl_executor.run()

    @classmethod
    @abc.abstractmethod
    def load_config(cls, config_path: Path) -> dict:
        raise NotImplementedError()

    @classmethod
    def entrypoint(cls):
        import typer

        typer.run(cls.run)

    @classmethod
    def get_fl_executor_from_config(cls, config: dict) -> FLExecutor:
        from fl_manager.core.schemas.fl_config import FLConfig

        _fl_config = FLConfig(**config)
        return _fl_config.fl_executor_instance

    @classmethod
    def to_script(cls) -> str:
        return (
            f'from {cls.__module__} import {cls.__name__}\n\n\n'
            f"if __name__ == '__main__':\n\t"
            f'{cls.__name__}.entrypoint()\n'
        )

    @classmethod
    def get_script_args(cls, config_path: str) -> list:
        return ['{WORKSPACE}', '{SITE_NAME}', '{JOB_ID}', f'custom/{config_path}']

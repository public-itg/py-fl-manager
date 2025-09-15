import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from fl_manager.core.schemas.registry_item import RegistryItem
from fl_manager.utils.job_generator.job_creator import JobCreator
from fl_manager.utils.job_generator.schemas.job_config import JobConfig


@pytest.fixture(scope='module')
def job_config():
    return JobConfig(
        name='mock_job',
        num_rounds=2,
        executor='mock_executor',
        executor_kwargs={},
        runner='mock_runner',
        components={
            'fl_model': RegistryItem(
                registry_id='fl_model',
                name='mock_fl_model',
                keyword_arguments={
                    'task': 'mock_task',
                    'key_metric': 'key_metric_value',
                    'negate_key_metric': False,
                },
            )
        },
        **{
            'fl_algorithm': {'name': 'fed_avg'},
            'clients': {'site-1': {}, 'site-2': {}},
        },
    )


def test_job_creator(
    register_mock_executor, register_mock_runner, register_mock_fl_model, job_config
):
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name)
    jobs_path = tmpdir_path / 'jobs'
    jc = JobCreator(job_config)
    jc.create(str(jobs_path))
    _job_dir = jobs_path / job_config.name
    assert _job_dir.exists()
    assert (server_path := (_job_dir / 'app_server')).exists()
    assert (site_1_path := (_job_dir / 'app_site-1')).exists()
    assert (site_2_path := (_job_dir / 'app_site-2')).exists()
    assert (server_path / 'config' / 'config_fed_server.json').stat().st_size > 0
    assert (site_1_path / 'config' / 'config_fed_client.json').stat().st_size > 0
    assert (site_2_path / 'config' / 'config_fed_client.json').stat().st_size > 0
    assert len(list((site_1_path / 'custom').iterdir())) != 0
    assert len(list((site_2_path / 'custom').iterdir())) != 0
    tmpdir.cleanup()


def test_no_clients(
    register_mock_executor, register_mock_runner, register_mock_fl_model, job_config
):
    jc = job_config.model_dump()
    with pytest.raises(ValidationError):
        JobConfig.model_validate(jc | {'clients': {}})

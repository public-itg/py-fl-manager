import pytest
from lightning import LightningModule

from fl_manager.lightning.dataflows.base_lightning_trainer import datamodule
from fl_manager.lightning.executors import (
    StatelessLightningFLExecutor,
    StatefulLightningFLExecutor,
)


@pytest.mark.parametrize(
    'fl_executor_cls', (StatelessLightningFLExecutor, StatefulLightningFLExecutor)
)
@pytest.mark.parametrize(
    'fl_config',
    (
        {'fl_algorithm': 'fed_avg', 'fl_algorithm_kwargs': {}},
        {'fl_algorithm': 'fed_bn', 'fl_algorithm_kwargs': {}},
        {'fl_algorithm': 'fed_opt', 'fl_algorithm_kwargs': {}},
        {'fl_algorithm': 'fed_prox', 'fl_algorithm_kwargs': {'mu': 0.1}},
    ),
)
def test_lightning_fl_executor(mocker, fl_executor_cls, fl_config, torch_dataset_alpha):
    mocker.patch(
        'hamilton.driver.Driver.execute',
        return_value={
            'datamodule': datamodule(
                dataset=torch_dataset_alpha,
                datamodule_name='default',
                datamodule_kwargs={},
            ),
            'model': LightningModule(),
        },
    )
    mocker.patch('nvflare.client.init')
    mocker.patch('nvflare.client.get_job_id')
    mocker.patch('nvflare.client.api.client_api', return_value='mocked_client_api')
    _fl_executor = fl_executor_cls(
        fl_train_id='mock-test', components={}, best_ckpt_kwargs={}, **fl_config
    )
    nv_executor_mock_start = mocker.patch.object(
        _fl_executor.nvflare_executor_cls, '_start_nvflare_executor', return_value=None
    )
    _fl_executor.run()
    nv_executor_mock_start.assert_called_once()

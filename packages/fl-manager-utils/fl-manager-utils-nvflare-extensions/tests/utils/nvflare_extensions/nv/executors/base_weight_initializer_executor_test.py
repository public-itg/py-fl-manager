import pytest
from nvflare.apis import dxo
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants

from fl_manager.utils.nvflare_extensions.nv.executors.base_weight_initializer_executor import (
    BaseWeightInitializerExecutor,
)


@pytest.fixture(scope='module')
def mock_weight_initializer_executor():
    class MockWeightInitializerExecutor(BaseWeightInitializerExecutor):
        def _get_model_weights(self) -> dict:
            return {'mock_layer_1': 1.0, 'mock_layer_2': 2.0}

    return MockWeightInitializerExecutor()


def test_weights_initializer_executor(mock_weight_initializer_executor):
    r = mock_weight_initializer_executor.execute(
        task_name=AppConstants.TASK_GET_WEIGHTS,
        shareable=Shareable(),
        **{'fl_ctx': None, 'abort_signal': None},
    )
    assert isinstance(r, Shareable)
    assert (
        dxo.from_shareable(r).data
        == mock_weight_initializer_executor._get_model_weights()
    )

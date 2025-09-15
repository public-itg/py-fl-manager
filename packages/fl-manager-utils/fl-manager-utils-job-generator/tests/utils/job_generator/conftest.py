from pathlib import Path
from typing import List

import pytest
import torch

from fl_manager.core.components.models import (
    FederatedLearningModel,
    FederatedLearningModelRegistry,
)
from fl_manager.core.executors import FLExecutorRegistry, FLExecutor
from fl_manager.core.runners import FLRunnerRegistry, FLRunner


@pytest.fixture(scope='module')
def register_mock_fl_model():
    @FederatedLearningModelRegistry.register(name='mock_fl_model')
    class MockFLModel(FederatedLearningModel):
        def __init__(self, task: str, key_metric: str, negate_key_metric: bool):
            super().__init__(
                task=task,
                key_metric=key_metric,
                negate_key_metric=negate_key_metric,
                model_cls=torch.nn.Module,
                seed=42,
            )

        @property
        def spec_supported_tasks(self) -> List[str]:
            return ['mock_task']

        def apply_seed(self):
            pass

        def get_weights(self) -> dict:
            return self.get_model().state_dict()

    return MockFLModel


@pytest.fixture(scope='module')
def register_mock_runner():
    @FLRunnerRegistry.register(name='mock_runner')
    class MockRunner(FLRunner):
        @classmethod
        def load_config(cls, config_path: Path):
            return {}

    return MockRunner


@pytest.fixture(scope='module')
def register_mock_executor():
    @FLExecutorRegistry.register(name='mock_executor')
    class MockExecutor(FLExecutor):
        def run(self):
            pass

    return MockExecutor

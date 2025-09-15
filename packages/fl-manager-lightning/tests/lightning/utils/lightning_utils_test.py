from copy import deepcopy

import pytest
from lightning import LightningModule
from torch import nn

from fl_manager.lightning.utils.lightning_utils import LightningUtils


class _A(LightningModule):
    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )


@pytest.fixture(scope='function')
def lightning_module():
    return _A()


@pytest.fixture(scope='module')
def another_lightning_module():
    return _A()


def test_get_different_keys(lightning_module, another_lightning_module):
    _state_dict = lightning_module.state_dict()
    _another_state_dict = another_lightning_module.state_dict()
    assert len(LightningUtils.get_different_keys(_state_dict, _state_dict)) == 0
    assert len(LightningUtils.get_different_keys(_state_dict, _another_state_dict)) != 0


def test_load_state_dict_no_skips(lightning_module, another_lightning_module):
    _prev_state_dict = deepcopy(lightning_module.state_dict())
    _new_state_dict = another_lightning_module.state_dict()
    assert (
        len(LightningUtils.get_different_keys(_prev_state_dict, _new_state_dict)) != 0
    )
    LightningUtils.load_state_dict_skipping_modules(
        pl_module=lightning_module, state_dict=_new_state_dict, strict=True
    )
    _updated_state_dict = lightning_module.state_dict()
    _diff_state_dict = LightningUtils.get_different_keys(
        _prev_state_dict, _updated_state_dict
    )
    assert (
        len(LightningUtils.get_different_keys(_new_state_dict, _updated_state_dict))
        == 0
    )
    assert len(_diff_state_dict) != 0
    assert any(['_model.0' in e for e in _diff_state_dict])


def test_load_state_dict_skipping_modules(lightning_module, another_lightning_module):
    _prev_state_dict = deepcopy(lightning_module.state_dict())
    LightningUtils.load_state_dict_skipping_modules(
        pl_module=lightning_module,
        state_dict=another_lightning_module.state_dict(),
        strict=False,
        modules_to_skip=[nn.Conv2d],
    )
    _updated_state_dict = lightning_module.state_dict()
    _diff_state_dict = LightningUtils.get_different_keys(
        _prev_state_dict, _updated_state_dict
    )
    assert len(_diff_state_dict) != 0
    assert any(['_model.0' not in e for e in _diff_state_dict])

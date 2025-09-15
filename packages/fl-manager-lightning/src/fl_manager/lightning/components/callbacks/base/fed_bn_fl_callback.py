from nvflare.app_opt.lightning import FLCallback
from torch import nn

from fl_manager.lightning.components.callbacks import LightningCallbackRegistry
from fl_manager.lightning.utils.lightning_callback_wrapper import (
    LightningCallbackWrapper,
)
from fl_manager.lightning.utils.lightning_utils import LightningUtils


class _FedBNFLCallback(FLCallback):
    def _receive_and_update_model(self, trainer, pl_module):
        model = self._receive_model(trainer)
        if model:
            if model.params:
                LightningUtils.load_state_dict_skipping_modules(
                    pl_module=pl_module,
                    state_dict=model.params,
                    strict=self._load_state_dict_strict,
                    modules_to_skip=[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d],
                )
            if model.current_round is not None:
                self.current_round = model.current_round


@LightningCallbackRegistry.register(name='fed_bn')
class FedBNFLCallback(LightningCallbackWrapper):
    def __init__(self, rank: int = 0, load_state_dict_strict: bool = True):
        super().__init__(callback=_FedBNFLCallback(rank, load_state_dict_strict))

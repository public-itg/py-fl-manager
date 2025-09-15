from nvflare.app_opt.lightning import FLCallback
from nvflare.app_opt.lightning.callbacks import RestoreState

from fl_manager.lightning.utils.lightning_callback_wrapper import (
    LightningCallbackWrapper,
)


class WrappedRestoreStateCallback(LightningCallbackWrapper):
    def __init__(self):
        super().__init__(callback=RestoreState())


class WrappedFLCallback(LightningCallbackWrapper):
    def __init__(self, rank: int = 0, load_state_dict_strict: bool = True):
        super().__init__(callback=FLCallback(rank, load_state_dict_strict))

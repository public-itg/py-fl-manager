from fl_manager.lightning.components.callbacks import LightningCallbackRegistry
from fl_manager.lightning.components.callbacks.nvflare_callbacks_wrappers import (
    WrappedFLCallback,
)


@LightningCallbackRegistry.register(name='fed_opt')
class FedOptFLCallback(WrappedFLCallback):
    pass

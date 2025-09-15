from fl_manager.lightning.components.callbacks import LightningCallbackRegistry
from fl_manager.lightning.components.callbacks.nvflare_callbacks_wrappers import (
    WrappedFLCallback,
)


@LightningCallbackRegistry.register(name='fed_avg')
class FedAvgFLCallback(WrappedFLCallback):
    pass

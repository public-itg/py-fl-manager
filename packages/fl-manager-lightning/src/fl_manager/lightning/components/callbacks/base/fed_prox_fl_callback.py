import copy

import pytorch_lightning as pl
from nvflare.app_opt.lightning import FLCallback
from nvflare.app_opt.pt import PTFedProxLoss
from torch import Tensor

from fl_manager.lightning.components.callbacks import LightningCallbackRegistry
from fl_manager.lightning.utils.lightning_callback_wrapper import (
    LightningCallbackWrapper,
)


class _FedProxFLCallback(FLCallback):
    def __init__(self, mu: float, rank: int = 0, load_state_dict_strict: bool = True):
        super().__init__(rank, load_state_dict_strict)
        self._mu = mu
        self._criterion_prox = PTFedProxLoss(mu=self._mu) if self._mu > 0 else None
        self._model_global = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # receive the global model and update the local model with global model
        self._receive_and_update_model(trainer, pl_module)
        # make a copy of model_global as reference for potential FedProx loss
        self._copy_model_global(pl_module)

    def on_before_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, loss: Tensor
    ) -> None:
        if self._criterion_prox is None:
            return
        if self._model_global is None:
            raise RuntimeError('Could not set reference model.')
        fed_prox_loss = self._criterion_prox(pl_module, self._model_global)
        # Modify loss directly before backward (used by Lightning)
        loss += fed_prox_loss

    def _copy_model_global(self, pl_module: pl.LightningModule):
        model_global = copy.deepcopy(pl_module)
        for param in model_global.parameters():
            param.requires_grad = False
        self._model_global = model_global


@LightningCallbackRegistry.register(name='fed_prox')
class FedProxFLCallback(LightningCallbackWrapper):
    def __init__(self, mu: float, rank: int = 0, load_state_dict_strict: bool = True):
        super().__init__(callback=_FedProxFLCallback(mu, rank, load_state_dict_strict))

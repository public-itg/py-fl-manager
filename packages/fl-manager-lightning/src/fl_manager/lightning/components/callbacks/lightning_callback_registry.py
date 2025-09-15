from lightning import Callback

from fl_manager.core.utils.registry import ClassRegistry

LightningCallbackRegistry = ClassRegistry[Callback](Callback)

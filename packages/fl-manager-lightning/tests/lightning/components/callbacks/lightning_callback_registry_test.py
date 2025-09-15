from lightning import Callback

from fl_manager.lightning.components.callbacks import LightningCallbackRegistry


def test_instances_class():
    assert all(
        [
            issubclass(LightningCallbackRegistry.get(e), Callback)
            for e in LightningCallbackRegistry.list()
        ]
    )

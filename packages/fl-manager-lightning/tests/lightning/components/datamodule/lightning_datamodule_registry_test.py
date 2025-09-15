from fl_manager.lightning.components.datamodule import LightningDataModuleRegistry
from fl_manager.lightning.components.datamodule.base_lightning_datamodule import (
    BaseLightningDataModule,
)


def test_instance_class():
    assert all(
        [
            issubclass(LightningDataModuleRegistry.get(e), BaseLightningDataModule)
            for e in LightningDataModuleRegistry.list()
        ]
    )

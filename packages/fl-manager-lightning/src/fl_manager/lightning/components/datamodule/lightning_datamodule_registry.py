from fl_manager.core.utils.registry import ClassRegistry
from fl_manager.lightning.components.datamodule.base_lightning_datamodule import (
    BaseLightningDataModule,
)

LightningDataModuleRegistry = ClassRegistry[BaseLightningDataModule](
    BaseLightningDataModule
)

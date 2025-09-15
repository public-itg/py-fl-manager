from fl_manager.lightning.components.datamodule import LightningDataModuleRegistry
from fl_manager.lightning.components.datamodule.base_lightning_datamodule import (
    BaseLightningDataModule,
)


@LightningDataModuleRegistry.register(name='default')
class DefaultDataModule(BaseLightningDataModule):
    @property
    def dataloader_kwargs(self) -> dict:
        return {}

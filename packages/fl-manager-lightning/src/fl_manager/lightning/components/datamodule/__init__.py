from fl_manager.core.utils.import_utils import ImportUtils

from fl_manager.lightning.components.datamodule.lightning_datamodule_registry import (
    LightningDataModuleRegistry,
)

__all__ = ['LightningDataModuleRegistry']

ImportUtils.dynamic_registry_item_import('datamodule', 'components')

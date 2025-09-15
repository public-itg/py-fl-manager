from fl_manager.core.utils.import_utils import ImportUtils
from fl_manager.lightning.components.callbacks.lightning_callback_registry import (
    LightningCallbackRegistry,
)

__all__ = ['LightningCallbackRegistry']

ImportUtils.dynamic_registry_item_import('callbacks', 'components')

from fl_manager.core.meta_registry import MetaRegistry
from fl_manager.core.utils.registry import ClassRegistry


def test_check_instance_class():
    assert all(
        [isinstance(MetaRegistry.get(e), ClassRegistry) for e in MetaRegistry.list()]
    )

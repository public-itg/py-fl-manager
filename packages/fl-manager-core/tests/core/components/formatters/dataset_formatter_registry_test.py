from fl_manager.core.components.formatters import (
    DatasetFormatterRegistry,
    DatasetFormatter,
)


def test_instances_class():
    assert all(
        [
            issubclass(DatasetFormatterRegistry.get(e), DatasetFormatter)
            for e in DatasetFormatterRegistry.list()
        ]
    )

from fl_manager.core.components.splitters import (
    DatasetSplitterRegistry,
    DatasetSplitter,
)


def test_instances_class():
    assert all(
        [
            issubclass(DatasetSplitterRegistry.get(e), DatasetSplitter)
            for e in DatasetSplitterRegistry.list()
        ]
    )

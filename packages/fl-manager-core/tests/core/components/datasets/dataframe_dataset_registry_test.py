from fl_manager.core.components.datasets import (
    DataFrameDatasetRegistry,
    DataFrameDataset,
)


def test_instances_class():
    assert all(
        [
            issubclass(DataFrameDatasetRegistry.get(e), DataFrameDataset)
            for e in DataFrameDatasetRegistry.list()
        ]
    )

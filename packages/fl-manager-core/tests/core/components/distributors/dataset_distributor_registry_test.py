from fl_manager.core.components.distributors import (
    DatasetDistributorRegistry,
    DatasetDistributor,
)


def test_instances_class():
    assert all(
        [
            issubclass(DatasetDistributorRegistry.get(e), DatasetDistributor)
            for e in DatasetDistributorRegistry.list()
        ]
    )

from fl_manager.core.components.readers import DatasetReaderRegistry, DatasetReader


def test_instances_class():
    assert all(
        [
            issubclass(DatasetReaderRegistry.get(e), DatasetReader)
            for e in DatasetReaderRegistry.list()
        ]
    )

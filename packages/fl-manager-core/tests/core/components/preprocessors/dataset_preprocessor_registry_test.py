from fl_manager.core.components.preprocessors import (
    DatasetPreprocessorRegistry,
    DatasetPreprocessor,
)


def test_instances_class():
    assert all(
        [
            issubclass(DatasetPreprocessorRegistry.get(e), DatasetPreprocessor)
            for e in DatasetPreprocessorRegistry.list()
        ]
    )

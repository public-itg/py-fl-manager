from fl_manager.core.components.validators import (
    DatasetValidatorRegistry,
    DatasetValidator,
)


def test_instances_class():
    assert all(
        [
            issubclass(DatasetValidatorRegistry.get(e), DatasetValidator)
            for e in DatasetValidatorRegistry.list()
        ]
    )

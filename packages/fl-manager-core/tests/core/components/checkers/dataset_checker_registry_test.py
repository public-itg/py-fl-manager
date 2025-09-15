from fl_manager.core.components.checkers import DatasetCheckerRegistry, DatasetChecker


def test_instances_class():
    assert all(
        [
            issubclass(DatasetCheckerRegistry.get(e), DatasetChecker)
            for e in DatasetCheckerRegistry.list()
        ]
    )

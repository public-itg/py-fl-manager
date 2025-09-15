from fl_manager.core.executors import FLExecutorRegistry, FLExecutor


def test_instances_class():
    assert all(
        [
            issubclass(FLExecutorRegistry.get(e), FLExecutor)
            for e in FLExecutorRegistry.list()
        ]
    )

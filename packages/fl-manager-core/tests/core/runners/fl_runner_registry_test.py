from fl_manager.core.runners import FLRunnerRegistry, FLRunner


def test_instances_class():
    assert all(
        [issubclass(FLRunnerRegistry.get(e), FLRunner) for e in FLRunnerRegistry.list()]
    )

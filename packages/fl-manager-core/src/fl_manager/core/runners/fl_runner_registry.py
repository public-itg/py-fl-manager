from fl_manager.core.runners.fl_runner import FLRunner
from fl_manager.core.utils.registry import ClassRegistry

FLRunnerRegistry = ClassRegistry[FLRunner](FLRunner)

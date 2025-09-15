from fl_manager.core.executors.fl_executor import FLExecutor
from fl_manager.core.utils.registry import ClassRegistry

FLExecutorRegistry = ClassRegistry[FLExecutor](FLExecutor)

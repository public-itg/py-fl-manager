import json
from pathlib import Path

from fl_manager.core.runners.fl_runner import FLRunner
from fl_manager.core.runners.fl_runner_registry import FLRunnerRegistry


@FLRunnerRegistry.register(name='base')
class BaseFLRunner(FLRunner):
    @classmethod
    def load_config(cls, config_path: Path):
        return json.loads(config_path.read_text())

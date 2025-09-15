import os
from pathlib import Path

PKG_NAME = 'fl-manager'

default_home = Path('~', '.cache').expanduser()
FL_MANAGER_HOME = Path(
    os.getenv('FL_MANAGER_HOME', default_home / PKG_NAME)
).expanduser()

FL_MANAGER_DATASET_CACHE = Path(
    os.getenv('FL_MANAGER_DATASET_CACHE', FL_MANAGER_HOME / 'datasets')
)

from pathlib import Path
from typing import TYPE_CHECKING

from fl_manager.core.components.readers import DatasetReaderRegistry, DatasetReader

if TYPE_CHECKING:
    from fl_manager.core.schemas.pandas_dataset import PandasDataset


@DatasetReaderRegistry.register(name='local')
class LocalDatasetReader(DatasetReader):
    """
    A reader for local datasets. Supports `.csv` files.

    - Registered in `DatasetReaderRegistry` with the name `local`.
    """

    def __init__(self, root_dir: str, dataset_filename: str):
        super().__init__()
        self._root_dir = Path(root_dir)
        self._dataset_path = self._root_dir / dataset_filename

    def fetch_dataset(self) -> 'PandasDataset':
        import pandas as pd
        from fl_manager.core.schemas.pandas_dataset import PandasDataset

        if not self._dataset_path.exists():
            raise FileNotFoundError(self._dataset_path)
        _extension = self._dataset_path.suffix
        match _extension:
            case '.csv':
                return PandasDataset(train=pd.read_csv(self._dataset_path))
            case _:
                raise TypeError(f'Unsupported file type ({_extension})!')

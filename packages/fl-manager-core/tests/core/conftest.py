import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fl_manager.core.components.splitters.base import ProportionDatasetSplitter
from fl_manager.core.schemas.pandas_dataset import PandasDataset


@pytest.fixture(scope='module')
def pandas_dataset_alpha():
    train_data = {
        'even_feature': range(0, 1000 * 2, 2),
        'odd_feature': range(1, 1000 * 2, 2),
        'label': [i % 10 for i in range(1000)],
    }
    _dataset = PandasDataset(train=pd.DataFrame(train_data))
    _distributor = ProportionDatasetSplitter()
    return _distributor.split(_dataset)


@pytest.fixture(scope='module')
def dummy_csv_dataset(pandas_dataset_alpha):
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name)
    csv_dataset = tmpdir_path / 'dataset.csv'
    _dataset = pandas_dataset_alpha.get_full_dataset().reset_index(drop=True)
    _dataset.to_csv(csv_dataset)
    yield csv_dataset
    tmpdir.cleanup()

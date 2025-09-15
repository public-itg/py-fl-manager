import pandas as pd
import pytest
from torch.utils.data import Dataset

from fl_manager.core.components.splitters.base import ProportionDatasetSplitter
from fl_manager.core.schemas.dataset import GenericDataset
from fl_manager.core.schemas.pandas_dataset import PandasDataset


@pytest.fixture(scope='module')
def torch_dataset_alpha_cls():
    class AlphaDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self._df = df

        def __len__(self):
            return len(self._df)

        def __getitem__(self, idx):
            return self._df.iloc[idx].values

    return AlphaDataset


@pytest.fixture(scope='module')
def pandas_dataset_alpha():
    train_data = {
        'even_feature': range(0, 1000 * 2, 2),
        'odd_feature': range(1, 1000 * 2, 2),
        'label': [i % 10 for i in range(1000)],
    }
    _distributor = ProportionDatasetSplitter()
    return _distributor.split(dataset=PandasDataset(train=pd.DataFrame(train_data)))


@pytest.fixture(scope='module')
def torch_dataset_alpha(pandas_dataset_alpha, torch_dataset_alpha_cls):
    return GenericDataset(
        train=torch_dataset_alpha_cls(pandas_dataset_alpha.train),
        val=torch_dataset_alpha_cls(pandas_dataset_alpha.val),
        test=torch_dataset_alpha_cls(pandas_dataset_alpha.test),
    )

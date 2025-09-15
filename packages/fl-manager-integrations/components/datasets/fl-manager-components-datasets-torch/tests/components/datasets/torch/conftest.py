import pandas as pd
import pytest

from fl_manager.core.components.preprocessors import DatasetPreprocessor
from fl_manager.core.components.splitters.base import ProportionDatasetSplitter
from fl_manager.core.schemas.pandas_dataset import PandasDataset


@pytest.fixture(scope='module')
def dataframe():
    data = {'feature': range(1000), 'label': [i % 10 for i in range(1000)]}
    return pd.DataFrame(data)


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
def preprocessor():
    class _A(DatasetPreprocessor):
        def preprocess(self, in_data: int) -> int:
            return (in_data + 1) * 2

    return _A()

import pandas as pd
import pytest

from fl_manager.core.schemas.pandas_dataset import PandasDataset


@pytest.fixture(scope='module')
def dataframe():
    data = {'feature': range(250), 'label': [i % 10 for i in range(250)]}
    return pd.DataFrame(data)


@pytest.fixture(scope='module')
def full_dataset(dataframe):
    return PandasDataset(train=dataframe, val=dataframe, test=dataframe)


@pytest.fixture(scope='module')
def only_train_dataset(dataframe):
    return PandasDataset(train=dataframe)


def test_full_dataset(full_dataset, dataframe):
    _full = full_dataset.get_full_dataset()
    assert len(_full) == (3 * len(dataframe))


def test_only_train_dataset(only_train_dataset, dataframe):
    _full = only_train_dataset.get_full_dataset()
    assert len(_full) == len(dataframe)

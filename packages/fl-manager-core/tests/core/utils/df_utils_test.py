import numpy as np
import pandas as pd
import pytest

from fl_manager.core.utils.df_utils import DFUtils


@pytest.fixture(scope='module')
def dataframe():
    data = {'feature': range(1000), 'label': [i % 10 for i in range(1000)]}
    return pd.DataFrame(data)


@pytest.fixture(scope='module')
def random_proportions_factory():
    def _make(n, decimals):
        scale = 10**decimals
        counts = np.random.multinomial(scale, np.ones(n) / n)
        proportions = counts / scale
        return proportions.tolist()

    return _make


def test_proportion_split(dataframe):
    r = DFUtils.proportion_split_dataframe(dataframe, [0.8, 0.1, 0.1])
    assert len(r) == 3
    assert sum([len(e) for e in r]) == len(dataframe)


def test_invalid_proportion_split(dataframe):
    with pytest.raises(AssertionError):
        DFUtils.proportion_split_dataframe(dataframe, [0.8, 0.2, 0.2])


@pytest.mark.parametrize('n', [3, 5, 7, 10])
@pytest.mark.parametrize('decimals', [4])
def test_random_proportion_split(dataframe, random_proportions_factory, n, decimals):
    props = random_proportions_factory(n, decimals=decimals)
    if (v := sum(props)) != 1.0:
        pytest.skip(f'Proportions do not sum up to 1.0, but {v}')
    r = DFUtils.proportion_split_dataframe(dataframe, props)
    assert len(r) == n
    assert sum([len(e) for e in r]) == len(dataframe)

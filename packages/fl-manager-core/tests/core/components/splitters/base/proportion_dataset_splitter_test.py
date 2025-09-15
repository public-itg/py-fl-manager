import pytest

from fl_manager.core.components.splitters.base import ProportionDatasetSplitter
from fl_manager.core.schemas.pandas_dataset import PandasDataset


@pytest.fixture(scope='module')
def proportion_dataset_splitter():
    return ProportionDatasetSplitter(proportions=[0.8, 0.1, 0.1])


def test_proportion_dataset_splitter_no_splits(
    pandas_dataset_alpha: PandasDataset, proportion_dataset_splitter
):
    assert all(
        [not e.empty for e in [pandas_dataset_alpha.val, pandas_dataset_alpha.test]]
    )
    _split_dataset = proportion_dataset_splitter.split(pandas_dataset_alpha)
    assert _split_dataset.train.equals(pandas_dataset_alpha.train)
    assert _split_dataset.val.equals(pandas_dataset_alpha.val)
    assert _split_dataset.test.equals(pandas_dataset_alpha.test)


def test_proportion_dataset_splitter_full_split(
    pandas_dataset_alpha: PandasDataset, proportion_dataset_splitter
):
    _split_dataset = proportion_dataset_splitter.split(
        PandasDataset(train=pandas_dataset_alpha.train)
    )
    assert len(_split_dataset.train) <= len(pandas_dataset_alpha.train)
    assert not _split_dataset.val.empty
    assert not _split_dataset.test.empty
    assert (
        _split_dataset.get_full_dataset()
        .reset_index(drop=True)
        .equals(pandas_dataset_alpha.train)
    )


def test_proportion_dataset_splitter_partial_split(
    pandas_dataset_alpha: PandasDataset, proportion_dataset_splitter
):
    _split_dataset_no_val = proportion_dataset_splitter.split(
        PandasDataset(train=pandas_dataset_alpha.train, test=pandas_dataset_alpha.test)
    )
    assert not _split_dataset_no_val.val.empty
    assert _split_dataset_no_val.test.equals(pandas_dataset_alpha.test)
    _split_dataset_no_test = proportion_dataset_splitter.split(
        PandasDataset(train=pandas_dataset_alpha.train, val=pandas_dataset_alpha.val)
    )
    assert _split_dataset_no_test.test.equals(pandas_dataset_alpha.val)

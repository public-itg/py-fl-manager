import pytest

from fl_manager.core.components.splitters.base import StratifiedDatasetSplitter
from fl_manager.core.schemas.pandas_dataset import PandasDataset


@pytest.fixture(scope='module')
def stratified_dataset_splitter():
    return StratifiedDatasetSplitter(
        target_col='label', min_samples_per_class=25, proportions=[0.8, 0.1, 0.1]
    )


def test_stratified_dataset_splitter_full_splits(
    pandas_dataset_alpha: PandasDataset, stratified_dataset_splitter
):
    _split_dataset = stratified_dataset_splitter.split(
        PandasDataset(train=pandas_dataset_alpha.train)
    )
    assert (
        _split_dataset.get_full_dataset()
        .sort_values('even_feature')
        .reset_index(drop=True)
        .equals(pandas_dataset_alpha.train)
    )

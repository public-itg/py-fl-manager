import pytest

from fl_manager.core.components.checkers.base import IsInstance
from fl_manager.core.components.formatters.base import ColumnDropperFormatter
from fl_manager.core.components.readers.base import LocalDatasetReader
from fl_manager.core.components.splitters.base import ProportionDatasetSplitter
from fl_manager.core.components.validators.base import BasePanderaDatasetValidator
from fl_manager.core.dataflows.dataset_setup import client_dataset, validated_dataset
from fl_manager.core.schemas.pandas_dataset import PandasDataset


@pytest.fixture(scope='module')
def dataset_reader(dummy_csv_dataset):
    return LocalDatasetReader(
        root_dir=str(dummy_csv_dataset.parent), dataset_filename=dummy_csv_dataset.name
    )


@pytest.fixture(scope='module')
def dataset_splitter():
    return ProportionDatasetSplitter(proportions=[0.8, 0.1, 0.1])


@pytest.fixture(scope='module')
def dataset_formatter():
    return ColumnDropperFormatter(in_col_name='even_feature')


@pytest.fixture(scope='module')
def dataset_validator():
    return BasePanderaDatasetValidator(
        columns={'odd_feature': [IsInstance(instance_cls=['int'])]}
    )


def test_dataset_setup(
    dataset_reader, dataset_splitter, dataset_formatter, dataset_validator
):
    _client_dataset = client_dataset(dataset_reader, dataset_splitter)
    assert isinstance(_client_dataset, PandasDataset)
    _validated_dataset = validated_dataset(
        _client_dataset, dataset_formatter, dataset_validator
    )
    assert isinstance(_validated_dataset, PandasDataset)

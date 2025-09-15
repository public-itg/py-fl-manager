import logging
from typing import Optional, List

from fl_manager.core.components.datasets import DataFrameDataset
from fl_manager.core.components.distributors import DatasetDistributor
from fl_manager.core.components.formatters import (
    DatasetFormatterRunner,
    DatasetFormatter,
)
from fl_manager.core.components.readers import DatasetReader
from fl_manager.core.components.splitters import DatasetSplitter
from fl_manager.core.components.validators import DatasetValidator
from fl_manager.core.schemas.dataset import GenericDataset
from fl_manager.core.schemas.pandas_dataset import PandasDataset

logger = logging.getLogger(__name__)


def client_dataset(
    dataset_reader: DatasetReader,
    dataset_splitter: DatasetSplitter,
    dataset_distributor: Optional[DatasetDistributor] = None,
) -> PandasDataset:
    _raw_dataset = dataset_reader.fetch_dataset()
    _client_dataset = _raw_dataset
    if dataset_distributor is not None:
        _client_dataset = dataset_distributor.get_dataset_distribution(_raw_dataset)
    client_split_dataset = dataset_splitter.split(_client_dataset)
    return client_split_dataset


def validated_dataset(
    client_dataset: PandasDataset,  # noqa
    dataset_formatter: Optional[DatasetFormatter | List[DatasetFormatter]] = None,
    dataset_validator: Optional[DatasetValidator] = None,
) -> PandasDataset:
    f_dataset = client_dataset
    if dataset_formatter is not None:
        f_dataset = DatasetFormatterRunner(dataset_formatter).run(client_dataset)
    if dataset_validator is not None:
        dataset_validator.validate(f_dataset)
    return f_dataset


def dataset(
    validated_dataset: PandasDataset,  # noqa
    dataframe_dataset: DataFrameDataset,
) -> GenericDataset:
    return dataframe_dataset.get_dataset(validated_dataset)

from typing import TYPE_CHECKING, Optional, Any

from fl_manager.core.components.readers import DatasetReader, DatasetReaderRegistry
from fl_manager.core.schemas.dataset import DatasetMapping

if TYPE_CHECKING:
    from pandas import DataFrame
    from fl_manager.core.schemas.pandas_dataset import PandasDataset


@DatasetReaderRegistry.register(name='huggingface')
class HuggingFaceDatasetReader(DatasetReader):
    def __init__(
        self,
        dataset_name: str,
        dataset_mapping: dict | DatasetMapping,
        dataset_subset: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        """
        Args:
            dataset_name: A local path or a dataset identifier from HuggingFaceHub.
            dataset_mapping: A dictionary with the relation of dataset split name (e.g. 'train', 'val', 'test').
            dataset_subset: Configuration name of the dataset, if any, from the HuggingFace dataset.
            trust_remote_code: Flag that enables remote code execution (if required).
        """
        from datasets import (
            get_dataset_split_names,
            get_dataset_config_names,
            get_dataset_default_config_name,
        )

        super().__init__(dataset_mapping)
        self._dataset_name = dataset_name
        self._load_dataset_kwargs: dict[str, Any] = {
            'trust_remote_code': trust_remote_code
        }
        self._config_name = dataset_subset or get_dataset_default_config_name(
            self._dataset_name
        )
        self._available_config_names = get_dataset_config_names(self._dataset_name)
        assert (
            self._config_name is not None
            and self._config_name in self._available_config_names
        )
        self._available_splits = get_dataset_split_names(
            self._dataset_name, self._config_name
        )

    def fetch_dataset(self) -> 'PandasDataset':
        """
        Returns:
             PandasDataset: Train, validation and test splits from the selected dataset at reader initialization.
        """
        from fl_manager.core.schemas.pandas_dataset import PandasDataset

        assert all(
            [
                split in self._available_splits
                for split in self.dataset_mapping.model_dump(exclude_none=True)
            ]
        ), f'invalid split name, choose one from {self._available_splits}'

        return PandasDataset(
            train=self._get_dataset_split_as_dataframe(self.dataset_mapping.train),
            val=self._get_dataset_split_as_dataframe(self.dataset_mapping.validation)
            if self.dataset_mapping.validation is not None
            else None,
            test=self._get_dataset_split_as_dataframe(self.dataset_mapping.test)
            if self.dataset_mapping.test
            else None,
        )

    def _get_dataset_split_as_dataframe(self, split: str) -> 'DataFrame':
        from datasets import load_dataset, Dataset
        from pandas import DataFrame

        _dataset = load_dataset(
            self._dataset_name,
            self._config_name,
            split=split,
            **self._load_dataset_kwargs,
        )
        assert isinstance(_dataset, Dataset)
        _dataframe = _dataset.to_pandas()
        assert isinstance(_dataframe, DataFrame)
        return _dataframe

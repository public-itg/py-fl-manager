import abc
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from fl_manager.core.component import Component
from fl_manager.core.utils.df_utils import DFUtils
from fl_manager.core.utils.nvflare_utils import NVFlareUtils

if TYPE_CHECKING:
    from pandas import DataFrame
    from fl_manager.core.schemas.pandas_dataset import PandasDataset


class DatasetDistributor(Component, metaclass=abc.ABCMeta):
    def __init__(
        self,
        num_clients: int,
        seed: Optional[int] = 42,
        with_server: Optional[bool] = True,
        global_test_size: Optional[float] = None,
    ):
        """
        Save a global test split and distribute remaining among clients.
        """
        assert num_clients >= 1, 'num_clients must be greater than or equal to 1'
        self._num_clients = num_clients
        self._seed = seed
        self._with_server = with_server
        _global_test_size = global_test_size or 0.1
        assert 0 <= _global_test_size <= 1, 'global_test_size must be between 0 and 1'
        self._global_test_size = _global_test_size
        self._rng = np.random.default_rng(self._seed)
        self._client_id = None
        self._global_test_split: Optional['DataFrame'] = None
        self._distributed_dataset: List['DataFrame'] | None = None

    @property
    def global_test_split(self) -> 'DataFrame':
        if self._global_test_split is None:
            raise ValueError('not distributed yet')
        return self._global_test_split

    @property
    def client_id(self) -> int:
        if self._client_id is None:
            if self._with_server:
                self._client_id = NVFlareUtils.get_client_id_for_data_distribution()
            else:
                self._client_id = self._rng.integers(0, self._num_clients)
            assert 0 <= self._client_id < self._num_clients, (
                f'client_id must be between 0 and {self._num_clients - 1}'
            )
        return self._client_id

    def get_dataset_distribution(self, dataset: 'PandasDataset') -> 'PandasDataset':
        from fl_manager.core.schemas.pandas_dataset import PandasDataset

        if self._distributed_dataset is None:
            self.apply_seed()
            _full_data = dataset.get_full_dataset()
            _data, self._global_test_split = DFUtils.proportion_split_dataframe(
                _full_data, [1 - self._global_test_size, self._global_test_size]
            )
            _distributed_dataset = self._distribute(_data)
            assert len(_distributed_dataset) == self._num_clients, (
                'non-matching number of distributions with number of clients'
            )
            self._distributed_dataset = _distributed_dataset
        return PandasDataset(train=self._distributed_dataset[self.client_id])

    @abc.abstractmethod
    def apply_seed(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _distribute(self, df: 'DataFrame') -> List['DataFrame']:
        raise NotImplementedError()

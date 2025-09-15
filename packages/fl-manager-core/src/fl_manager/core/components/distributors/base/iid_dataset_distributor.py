from typing import TYPE_CHECKING, List

import numpy as np

from fl_manager.core.components.distributors import (
    DatasetDistributor,
    DatasetDistributorRegistry,
)

if TYPE_CHECKING:
    from pandas import DataFrame


@DatasetDistributorRegistry.register(name='iid')
class IIDDatasetDistributor(DatasetDistributor):
    def apply_seed(self):
        np.random.seed(self._seed)

    def _distribute(self, df: 'DataFrame') -> List['DataFrame']:
        _df = df.sample(frac=1).reset_index().rename(columns={'index': '_src_index'})
        dist = np.array_split(_df.index, self._num_clients)
        return [_df.iloc[split].reset_index(drop=True) for split in dist]

from typing import TYPE_CHECKING

import numpy as np

from fl_manager.core.components.splitters import (
    DatasetSplitterRegistry,
    BaseProportionDatasetSplitter,
)
from fl_manager.core.utils.df_utils import DFUtils

if TYPE_CHECKING:
    from pandas import DataFrame
    from fl_manager.core.schemas.pandas_dataset import PandasDataset


@DatasetSplitterRegistry.register(name='proportion')
class ProportionDatasetSplitter(BaseProportionDatasetSplitter):
    def apply_seed(self):
        np.random.seed(self._seed)

    def _split_dataframe(self, df: 'DataFrame') -> 'PandasDataset':
        from fl_manager.core.schemas.pandas_dataset import PandasDataset

        _keys = PandasDataset.model_fields.keys()  # noqa
        _splits = DFUtils.proportion_split_dataframe(df, self._proportions)
        return PandasDataset(**dict(zip(_keys, _splits, strict=True)))

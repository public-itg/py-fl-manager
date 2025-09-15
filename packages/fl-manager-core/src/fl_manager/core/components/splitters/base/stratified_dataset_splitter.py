import logging
from typing import TYPE_CHECKING, Optional, List, cast

import numpy as np

from fl_manager.core.components.splitters import (
    BaseProportionDatasetSplitter,
    DatasetSplitterRegistry,
)

if TYPE_CHECKING:
    from pandas import DataFrame
    from fl_manager.core.schemas.pandas_dataset import PandasDataset

logger = logging.getLogger(__name__)


@DatasetSplitterRegistry.register(name='stratified')
class StratifiedDatasetSplitter(BaseProportionDatasetSplitter):
    def __init__(
        self,
        target_col: str,
        min_samples_per_class: Optional[int] = 200,
        proportions: Optional[List[float]] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__(proportions, seed)
        self._target_col = target_col
        self._min_samples_per_class = min_samples_per_class

    def apply_seed(self):
        np.random.seed(self._seed)

    def _split_dataframe(self, df: 'DataFrame') -> 'PandasDataset':
        from pandas import DataFrame, Series
        from sklearn.model_selection import train_test_split
        from fl_manager.core.schemas.pandas_dataset import PandasDataset

        assert self._target_col in df.columns, (
            'components does not contain target column.'
        )
        _target_counts = df[self._target_col].value_counts()
        rare_targets = cast(
            Series, _target_counts[_target_counts < self._min_samples_per_class]
        ).index.tolist()
        if rare_targets:
            logger.warning(
                f'Ignoring rare targets {rare_targets} (less than {self._min_samples_per_class} samples).'
            )
            _df = df[~df[self._target_col].isin(rare_targets)]
        else:
            _df = df
        _, val_size, test_size = self._proportions
        train_df, temp_df = train_test_split(
            _df,
            test_size=(val_size + test_size),
            stratify=_df[self._target_col],
            random_state=self._seed,
        )
        assert isinstance(temp_df, DataFrame)
        # Split temp into val and test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(test_size / (val_size + test_size)),
            stratify=temp_df[self._target_col],
            random_state=self._seed,
        )
        assert isinstance(train_df, DataFrame)
        assert isinstance(val_df, DataFrame)
        assert isinstance(test_df, DataFrame)
        return PandasDataset(train=train_df, val=val_df, test=test_df)

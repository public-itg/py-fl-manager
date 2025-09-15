import abc
import logging
from typing import TYPE_CHECKING, Optional, List

from fl_manager.core.component import Component

if TYPE_CHECKING:
    from pandas import DataFrame
    from fl_manager.core.schemas.pandas_dataset import PandasDataset

logger = logging.getLogger(__name__)


class DatasetSplitter(Component, metaclass=abc.ABCMeta):
    def __init__(self, seed: Optional[int] = 42):
        self._seed = seed

    def split(self, dataset: 'PandasDataset') -> 'PandasDataset':
        self.apply_seed()
        _split = self._split(dataset)
        _split.train.reset_index(drop=True, inplace=True)
        if _split.val is not None:
            _split.val.reset_index(drop=True, inplace=True)
        if _split.test is not None:
            _split.test.reset_index(drop=True, inplace=True)
        return _split

    @abc.abstractmethod
    def _split(self, dataset: 'PandasDataset') -> 'PandasDataset':
        raise NotImplementedError()

    @abc.abstractmethod
    def apply_seed(self):
        raise NotImplementedError()


class BaseProportionDatasetSplitter(DatasetSplitter, metaclass=abc.ABCMeta):
    def __init__(
        self, proportions: Optional[List[float]] = None, seed: Optional[int] = 42
    ):
        super().__init__(seed)
        proportions = proportions or [0.8, 0.1, 0.1]
        self._proportions_sanity_check(proportions)
        self._proportions = proportions

    def _split(self, dataset: 'PandasDataset') -> 'PandasDataset':
        """
        The split mapping may present four cases:
            (1): Only train components is available.
            (2): Train components and validation components is available.
            (3): Train components and test components is available.
            (4): All three splits are available.
        The split mapping is defined by the user, so we assume that the DatasetReader can read these splits accordingly.
        The proportions configured in the DatasetSplitter is according the available splits as well.
        The resolution uses the DatasetSplitter following the following strategies:
            (1): Use train to generate train, validation, and test splits.
            (2): (unusual case), validation components is the test components and use train to generate train and validation splits.
            (3): Use train to generate train and validation splits.
            (4): No further actions required.
        """
        from fl_manager.core.schemas.pandas_dataset import PandasDataset

        exists_val = dataset.val is not None
        exists_test = dataset.test is not None
        if exists_val and exists_test:
            return dataset  # Resolution (4)
        if not exists_val and not exists_test:
            return self._split_dataframe(dataset.train)  # Resolution (1)
        if not exists_test:
            logger.warning(
                'Defined train and validation splits, but no test split is available. '
                'Using validation and test, and splitting train for train and validation (new).'
            )
        self._readjust_proportions([True, True, False])
        _splits = self._split_dataframe(dataset.train)  # Resolution (2) and (3)
        return PandasDataset(
            train=_splits.train,
            val=_splits.val,
            test=dataset.test if exists_test else dataset.val,
        )

    @abc.abstractmethod
    def _split_dataframe(self, df: 'DataFrame') -> 'PandasDataset':
        raise NotImplementedError()

    def _readjust_proportions(self, removal_mask: List[bool]) -> List[float]:
        assert sum(removal_mask) < 3, 'Cannot mask out all proportions'
        assert len(removal_mask) == 3, 'Mask must have length 3 (train, val, test)'
        _masked_proportions = [
            p if m else 0 for p, m in zip(self._proportions, removal_mask, strict=True)
        ]
        _new_total = sum(_masked_proportions)
        _readjusted_proportions = [
            round(p / _new_total, 2) for p in _masked_proportions
        ]
        self._proportions_sanity_check(_readjusted_proportions)
        logging.info(
            f'Readjusted proportions from {self._proportions} to {_readjusted_proportions}.'
        )
        self._proportions = _readjusted_proportions
        return self._proportions

    @staticmethod
    def _proportions_sanity_check(proportions: List[float]) -> None:
        assert sum(proportions) == 1, 'Proportions must sum to 1'
        assert len(proportions) == 3, (
            'Proportions must have length 3 (train, val, test) proportions'
        )

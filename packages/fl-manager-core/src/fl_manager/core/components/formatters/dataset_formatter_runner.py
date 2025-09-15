from typing import TYPE_CHECKING, List

from fl_manager.core.components.formatters import (
    DatasetFormatter,
    DatasetFormatterComposite,
)
from fl_manager.core.utils.composite_utils import CompositeUtils

if TYPE_CHECKING:
    from fl_manager.core.schemas.pandas_dataset import PandasDataset


class DatasetFormatterRunner:
    def __init__(self, dataset_formatter: DatasetFormatter | List[DatasetFormatter]):
        self._dataset_formatter = (
            CompositeUtils.leafs_to_composite(
                composite=DatasetFormatterComposite(), leafs=dataset_formatter
            )
            if isinstance(dataset_formatter, list)
            else dataset_formatter
        )

    def run(self, dataset: 'PandasDataset') -> 'PandasDataset':
        from fl_manager.core.schemas.pandas_dataset import PandasDataset

        return PandasDataset(
            train=dataset.train
            if dataset.train.empty
            else self._dataset_formatter.run(dataset.train),
            val=self._dataset_formatter.run(dataset.val)
            if dataset.val is not None and not dataset.val.empty
            else dataset.val,
            test=self._dataset_formatter.run(dataset.test)
            if dataset.test is not None and not dataset.test.empty
            else dataset.test,
        )

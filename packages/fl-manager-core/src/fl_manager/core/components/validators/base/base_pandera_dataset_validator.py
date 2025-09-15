from typing import TYPE_CHECKING, List, Dict

from fl_manager.core.components.checkers import DatasetChecker
from fl_manager.core.components.validators import (
    DatasetValidator,
    DatasetValidatorRegistry,
)

if TYPE_CHECKING:
    from fl_manager.core.schemas.pandas_dataset import PandasDataset


@DatasetValidatorRegistry.register(name='base_pandera')
class BasePanderaDatasetValidator(DatasetValidator):
    def __init__(self, columns: Dict[str, List[DatasetChecker]]):
        import pandera.pandas as pa

        self._columns = {
            k: pa.Column(checks=[c.get_checker() for c in v])
            for k, v in columns.items()
        }
        self._dataframe_schema = pa.DataFrameSchema(columns=self._columns)

    def validate(self, in_data: 'PandasDataset'):
        self._dataframe_schema.validate(in_data.train)
        if in_data.val is not None:
            self._dataframe_schema.validate(in_data.val)
        if in_data.test is not None:
            self._dataframe_schema.validate(in_data.test)

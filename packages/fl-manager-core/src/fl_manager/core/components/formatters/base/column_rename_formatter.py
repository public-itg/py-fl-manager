from typing import TYPE_CHECKING

from fl_manager.core.components.formatters import (
    DatasetFormatterRegistry,
    BaseDatasetFormatter,
)

if TYPE_CHECKING:
    from pandas import DataFrame


@DatasetFormatterRegistry.register(name='column_rename')
class ColumnRenameFormatter(BaseDatasetFormatter):
    def _run_formatter(self, in_data: 'DataFrame') -> 'DataFrame':
        assert self._out_col_name is not None, 'missing output column name'
        assert self._out_col_name not in in_data.columns, 'column already exists'
        return in_data.rename(columns={self._in_col_name: self._out_col_name})

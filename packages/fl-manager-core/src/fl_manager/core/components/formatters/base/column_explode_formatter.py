from typing import TYPE_CHECKING, Optional

from fl_manager.core.components.formatters import (
    BaseDatasetFormatter,
    DatasetFormatterRegistry,
)

if TYPE_CHECKING:
    from pandas import DataFrame


@DatasetFormatterRegistry.register(name='column_explode')
class ColumnExplodeFormatter(BaseDatasetFormatter):
    def __init__(
        self, in_col_name: str, extra_in_col_names: Optional[tuple[str]] = None
    ):
        super().__init__(in_col_name)
        self._to_explode_col_names = self._in_col_name, *(extra_in_col_names or ())

    def _run_formatter(self, in_data: 'DataFrame') -> 'DataFrame':
        return in_data.explode(column=list(self._to_explode_col_names))

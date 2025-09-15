from typing import TYPE_CHECKING, Optional

from fl_manager.core.components.formatters import (
    DatasetFormatterRegistry,
    BaseDatasetFormatter,
)

if TYPE_CHECKING:
    from pandas import DataFrame


@DatasetFormatterRegistry.register(name='column_dropper')
class ColumnDropperFormatter(BaseDatasetFormatter):
    def __init__(
        self, in_col_name: str, extra_in_col_names: Optional[tuple[str]] = None
    ):
        super().__init__(in_col_name)
        self._to_drop_col_names = self._in_col_name, *(extra_in_col_names or ())

    def _run_formatter(self, in_data: 'DataFrame') -> 'DataFrame':
        current_col_names = in_data.columns.tolist()
        out_col_names = set(current_col_names).intersection(self._to_drop_col_names)
        return in_data.drop(columns=list(out_col_names))

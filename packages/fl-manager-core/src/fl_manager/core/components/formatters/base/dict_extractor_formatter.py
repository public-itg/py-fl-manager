from typing import TYPE_CHECKING, List

from fl_manager.core.components.formatters import (
    DatasetFormatterRegistry,
    BaseDatasetFormatter,
)

if TYPE_CHECKING:
    from pandas import DataFrame


@DatasetFormatterRegistry.register(name='dict_extractor')
class DictExtractorFormatter(BaseDatasetFormatter):
    def __init__(self, in_col_name: str, keys: List[str]):
        super().__init__(in_col_name, out_col_name=None)
        self._keys = keys

    def _run_formatter(self, in_data: 'DataFrame') -> 'DataFrame':
        import pandas as pd

        _normalized_df = pd.json_normalize(in_data[self._in_col_name].to_list())
        if _normalized_df.empty:
            raise ValueError(f'Not found json structure: {self._in_col_name}.')
        _columns = _normalized_df.columns.tolist()
        _keys = [k for k in self._keys if k in _columns]
        if not _keys:
            raise ValueError('No matching keys.')
        _normalized_df = _normalized_df[_keys]
        return pd.concat(
            [in_data, _normalized_df.add_prefix(f'{self._in_col_name}__')], axis=1
        )

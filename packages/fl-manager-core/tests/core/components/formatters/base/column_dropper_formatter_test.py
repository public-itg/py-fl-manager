from fl_manager.core.components.formatters.base import ColumnDropperFormatter


def test_column_dropper(pandas_dataset_alpha):
    formatter = ColumnDropperFormatter(
        in_col_name='even_feature', extra_in_col_names=('odd_feature',)
    )
    _formatted = formatter.run(pandas_dataset_alpha.train)
    assert 'even_feature' not in _formatted.columns
    assert 'odd_feature' not in _formatted.columns

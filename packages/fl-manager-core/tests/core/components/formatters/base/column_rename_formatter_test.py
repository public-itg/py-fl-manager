import pytest

from fl_manager.core.components.formatters.base import ColumnRenameFormatter


def test_column_rename_formatter(pandas_dataset_alpha):
    formatter = ColumnRenameFormatter(
        in_col_name='odd_feature', out_col_name='renamed_odd_feature'
    )
    _formatted = formatter.run(pandas_dataset_alpha.train)
    assert 'renamed_odd_feature' in _formatted.columns


def test_existent_output_column(pandas_dataset_alpha):
    formatter = ColumnRenameFormatter(
        in_col_name='odd_feature', out_col_name='even_feature'
    )
    with pytest.raises(AssertionError) as ex:
        formatter.run(pandas_dataset_alpha.train)
    assert 'column already exists' in str(ex)


def test_missing_output_column(pandas_dataset_alpha):
    formatter = ColumnRenameFormatter(in_col_name='odd_feature')
    with pytest.raises(AssertionError) as ex:
        formatter.run(pandas_dataset_alpha.train)
    assert 'missing output column name' in str(ex)

import pandas as pd
import pytest

from fl_manager.core.components.formatters.base import DictExtractorFormatter


@pytest.fixture
def dataframe_with_dict():
    data = {
        'dict_feature': [{'value': v, 'value_p2': v * 2} for v in range(1000)],
        'label': [i % 10 for i in range(1000)],
    }
    return pd.DataFrame(data)


def test_dict_extractor_formatter(dataframe_with_dict):
    formatter = DictExtractorFormatter(
        in_col_name='dict_feature', keys=['value', 'value_p2']
    )
    _formatted = formatter.run(dataframe_with_dict)
    assert (
        len(_formatted[['dict_feature__value', 'dict_feature__value_p2']].columns) == 2
    )


def test_no_matching_keys_dict_extractor_formatter(dataframe_with_dict):
    formatter = DictExtractorFormatter(
        in_col_name='dict_feature', keys=['non_existent']
    )
    with pytest.raises(ValueError):
        formatter.run(dataframe_with_dict)

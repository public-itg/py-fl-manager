import pytest

from fl_manager.core.schemas.base_variables import BaseVariables


@pytest.fixture(scope='module')
def json_config():
    return {
        'variables': {'v1': 'v1 value'},
        'data': {'any_name': 'any value', 'v1': '{{v1}}'},
    }


@pytest.fixture(scope='module')
def data_class():
    class _A(BaseVariables):
        data: dict

    return _A


def test_base_variables(json_config, data_class):
    r = data_class.model_validate(json_config)
    assert r.data.get('v1') == 'v1 value'


def test_missing_variables(json_config, data_class):
    _json_config = json_config.copy()
    _json_config['variables'] = {}
    with pytest.raises(ValueError) as ex:
        data_class.model_validate(_json_config)
    assert 'missing variables' in str(ex) and 'v1' in str(ex)


def test_non_match_variable_format(json_config, data_class):
    _json_config = json_config.copy()
    _json_config['data']['v1'] = '{v1}'
    r = data_class.model_validate(_json_config)
    assert r.data.get('v1') == '{v1}'

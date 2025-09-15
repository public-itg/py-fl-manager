import pytest

from fl_manager.core.utils.nvflare_utils import NVFlareUtils


def test_get_client_id(mocker):
    mock_flare_init = mocker.patch('nvflare.client.init')
    mock_flare_get_site_name = mocker.patch('nvflare.client.get_site_name')
    mock_flare_get_site_name.return_value = 'site_name_2'
    client_id = NVFlareUtils.get_client_id_for_data_distribution()
    assert client_id == 1
    mock_flare_init.assert_called_once()
    mock_flare_get_site_name.assert_called_once()
    mock_flare_get_site_name.return_value = 'site_name'
    with pytest.raises(AssertionError):
        NVFlareUtils.get_client_id_for_data_distribution()
    mock_flare_get_site_name.return_value = 'site_name_0'
    with pytest.raises(AssertionError):
        NVFlareUtils.get_client_id_for_data_distribution()

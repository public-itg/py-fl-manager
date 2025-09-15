import logging
import re

logger = logging.getLogger(__name__)


class NVFlareUtils:
    @staticmethod
    def get_client_id_for_data_distribution() -> int:
        """
        Auxiliary method to get the client id for components distribution (simulating FL scenarios).
        Site name must end with 'number' and be greater than 0.
        """
        import nvflare.client as flare

        flare.init()
        _site_name = flare.get_site_name()
        _pattern = re.compile(r'\d{1,100000}$')
        match = _pattern.search(_site_name)
        assert match is not None, (
            'invalid site name, for data_distribution the site name must end with a "number".'
        )
        _client_id = int(match.group())
        assert _client_id > 0, 'the site number must be greater than 0.'
        return _client_id - 1

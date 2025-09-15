from fl_manager.core.components.distributors.base import IIDDatasetDistributor
from fl_manager.core.schemas.pandas_dataset import PandasDataset


def test_iid_dataset_distributor(pandas_dataset_alpha):
    distributor = IIDDatasetDistributor(num_clients=3, with_server=False)
    _distributed = distributor.get_dataset_distribution(pandas_dataset_alpha)
    assert isinstance(_distributed, PandasDataset)
    assert len(distributor._distributed_dataset) == 3

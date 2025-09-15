from fl_manager.core.components.distributors.base import DirichletDatasetDistributor
from fl_manager.core.schemas.pandas_dataset import PandasDataset


def test_dirichlet_dataset_distributor(pandas_dataset_alpha):
    distributor = DirichletDatasetDistributor(
        num_clients=3, target_col='label', alpha=0.3, with_server=False
    )
    _distributed = distributor.get_dataset_distribution(pandas_dataset_alpha)
    assert isinstance(_distributed, PandasDataset)
    assert len(distributor._distributed_dataset) == 3

from fl_manager.components.datasets.torch import TorchDataFrameTransformsDataset
from fl_manager.core.components.datasets import DataFrameDatasetRegistry
from fl_manager.core.schemas.dataset import GenericDataset


def test_torch_dataframe_transform_dataset_registered():
    assert 'torch_dataframe_transforms_dataset' in DataFrameDatasetRegistry.list()


def test_torch_dataframe_transform_dataset(pandas_dataset_alpha, preprocessor):
    dataframe_dataset = TorchDataFrameTransformsDataset(
        {'odd_feature': preprocessor}, output_keys=['odd_feature'], sample_as_dict=True
    )
    dataset = dataframe_dataset.get_dataset(pandas_dataset_alpha)
    assert isinstance(dataset, GenericDataset)
    _sample = dataset.train[0]
    assert isinstance(_sample, dict)
    assert 'odd_feature' in _sample

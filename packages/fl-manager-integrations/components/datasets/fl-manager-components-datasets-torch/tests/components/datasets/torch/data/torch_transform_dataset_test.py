import pytest

from fl_manager.components.datasets.torch.data.torch_transforms_dataset import (
    TorchTransformsDataset,
)


def test_transform_dataset(dataframe, preprocessor):
    dataset = TorchTransformsDataset(df=dataframe, transforms={'feature': preprocessor})
    assert len(dataset) == len(dataframe)
    dataset_2 = TorchTransformsDataset(df=dataframe)
    assert (dataset[0] == dataset[0]) and (dataset_2[0] == dataset_2[0])
    assert dataset[0] != dataset_2[0]


def test_invalid_output_keys(dataframe):
    with pytest.raises(AssertionError) as ex:
        TorchTransformsDataset(df=dataframe, output_keys=['non_existent_key'])
    assert 'invalid output keys' in str(ex.value)

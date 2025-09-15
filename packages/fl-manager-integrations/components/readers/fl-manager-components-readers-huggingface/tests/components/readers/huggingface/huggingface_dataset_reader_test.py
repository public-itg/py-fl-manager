import pytest
from datasets import Dataset

from fl_manager.components.readers.huggingface import HuggingFaceDatasetReader
from fl_manager.core.schemas.dataset import DatasetMapping
from fl_manager.core.schemas.pandas_dataset import PandasDataset


@pytest.fixture(scope='module')
def fake_dataset():
    data = [{'text': 'Good movie!', 'label': 1}, {'text': 'Bad movie.', 'label': 0}]
    return Dataset.from_list(data)


def test_huggingface_dataset_reader(mocker, fake_dataset):
    mocker.patch('datasets.get_dataset_default_config_name', return_value='default')
    mocker.patch('datasets.get_dataset_config_names', return_value='default')
    mocker.patch('datasets.get_dataset_split_names', return_value=['train'])
    mocker.patch('datasets.load_dataset', return_value=fake_dataset)
    reader = HuggingFaceDatasetReader(
        dataset_name='hf-internal-testing/fill10',
        dataset_mapping=DatasetMapping(train='train'),
    )
    dataset = reader.fetch_dataset()
    assert isinstance(dataset, PandasDataset)
    assert not dataset.train.empty
    assert dataset.val is None
    assert dataset.test is None

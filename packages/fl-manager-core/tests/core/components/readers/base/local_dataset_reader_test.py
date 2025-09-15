from fl_manager.core.components.readers.base import LocalDatasetReader


def test_local_dataset_reader(dummy_csv_dataset, pandas_dataset_alpha):
    reader = LocalDatasetReader(
        root_dir=str(dummy_csv_dataset.parent), dataset_filename=dummy_csv_dataset.name
    )
    dataset = reader.fetch_dataset()
    _dataset = pandas_dataset_alpha.get_full_dataset().reset_index(drop=True)
    dataset = dataset.get_full_dataset()[list(_dataset.columns)]
    assert dataset.equals(_dataset)

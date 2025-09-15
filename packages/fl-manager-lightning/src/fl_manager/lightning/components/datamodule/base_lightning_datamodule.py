import abc
from typing import Optional, Union, Iterable, Callable

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from lightning_utilities import apply_to_collection
from torch.utils.data import Dataset, DataLoader, IterableDataset


class BaseLightningDataModule(LightningDataModule, metaclass=abc.ABCMeta):
    def __init__(
        self, batch_size: int = 1, num_workers: int = 0, pin_memory: bool = True
    ):
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._train_dataloader: Callable | None = None
        self._val_dataloader: Callable | None = None
        self._test_dataloader: Callable | None = None
        self._predict_dataloader: Callable | None = None

    @property
    @abc.abstractmethod
    def dataloader_kwargs(self) -> dict:
        raise NotImplementedError()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self._train_dataloader is None:
            return super().train_dataloader()
        return self._train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self._val_dataloader is None:
            return super().val_dataloader()
        return self._val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self._test_dataloader is None:
            return super().test_dataloader()
        return self._test_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self._predict_dataloader is None:
            return super().predict_dataloader()
        return self._predict_dataloader()

    def configure_dataloaders_from_datasets(
        self,
        train_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
    ):
        def train_dataloader() -> TRAIN_DATALOADERS:
            return apply_to_collection(
                train_dataset, Dataset, self._dataloader, shuffle=True
            )

        def val_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(val_dataset, Dataset, self._dataloader)

        def test_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(test_dataset, Dataset, self._dataloader)

        def predict_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(predict_dataset, Dataset, self._dataloader)

        if train_dataset is not None:
            self._train_dataloader = train_dataloader
        if val_dataset is not None:
            self._val_dataloader = val_dataloader
        if test_dataset is not None:
            self._test_dataloader = test_dataloader
        if predict_dataset is not None:
            self._predict_dataloader = predict_dataloader

    def _dataloader(self, ds: Dataset, shuffle: bool = False) -> DataLoader:
        shuffle &= not isinstance(ds, IterableDataset)
        _dataloader_kwargs = self.dataloader_kwargs | {
            'batch_size': self._batch_size,
            'shuffle': shuffle,
            'num_workers': self._num_workers,
            'pin_memory': self._pin_memory,
        }
        return DataLoader(ds, **_dataloader_kwargs)

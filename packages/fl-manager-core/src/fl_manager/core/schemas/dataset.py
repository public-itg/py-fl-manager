from typing import Optional, Any

from pydantic import BaseModel


class DatasetMapping(BaseModel):
    """Define the split naming equivalence in the dataset to the corresponding split (if any)."""

    train: str
    validation: Optional[str] = None
    test: Optional[str] = None


class GenericDataset(BaseModel, arbitrary_types_allowed=True):
    train: Any
    val: Optional[Any] = None
    test: Optional[Any] = None

    def __str__(self):
        _n_val = len(self.val) if self.val is not None else 0
        _n_test = len(self.test) if self.test is not None else 0
        return f'{self.__class__.__name__}(train={len(self.train)}, val={_n_val}, test={_n_test})'

    def __repr__(self):
        return self.__str__()

    def get_full_dataset(self):
        raise NotImplementedError()

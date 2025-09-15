import logging
from typing import Tuple, Dict, Optional, List

import pandas as pd
from torch.utils.data import Dataset

from fl_manager.core.components.preprocessors import DatasetPreprocessor

logger = logging.getLogger(__name__)


class TorchTransformsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Optional[Dict[str, DatasetPreprocessor]] = None,
        output_keys: Optional[List[str]] = None,
        sample_as_dict: Optional[bool] = False,
    ):
        _all_data_keys = df.columns.to_list()
        self._output_keys = output_keys or _all_data_keys
        assert set(self._output_keys).issubset(_all_data_keys), 'invalid output keys'
        self._df = df[self._output_keys]
        _transforms = transforms or {}
        self.transforms = {
            k: v for k, v in _transforms.items() if k in self._output_keys
        }
        if len(_transforms) != len(self.transforms):
            logger.warning(
                'Some transforms were not applied, not present in the desired output'
            )
        self.sample_as_dict = sample_as_dict

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple | Dict:
        return self._return_sample(
            sample=self._sample_transform(self._df.iloc[idx].to_dict())
        )

    def _sample_transform(self, sample: dict) -> dict:
        for key, pipeline in self.transforms.items():
            sample[key] = pipeline.preprocess(sample[key])
        return sample

    def _return_sample(self, sample: dict) -> Tuple | Dict:
        return sample if self.sample_as_dict else tuple(sample.values())

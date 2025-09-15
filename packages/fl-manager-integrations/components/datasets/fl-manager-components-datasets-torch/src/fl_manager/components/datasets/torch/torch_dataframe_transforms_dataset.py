from typing import Dict, Optional, List, Union

from fl_manager.core.components.datasets import (
    DataFrameDatasetRegistry,
    DataFrameDataset,
)
from fl_manager.core.components.preprocessors import (
    DatasetPreprocessor,
    DatasetPreprocessorComposite,
)
from fl_manager.core.utils.composite_utils import CompositeUtils


@DataFrameDatasetRegistry.register(name='torch_dataframe_transforms_dataset')
class TorchDataFrameTransformsDataset(DataFrameDataset):
    def __init__(
        self,
        transforms: Optional[
            Dict[str, Union[DatasetPreprocessor, List[DatasetPreprocessor]]]
        ] = None,
        output_keys: Optional[List[str]] = None,
        sample_as_dict: Optional[bool] = False,
    ):
        from .data.torch_transforms_dataset import TorchTransformsDataset

        _transforms = {
            k: CompositeUtils.leafs_to_composite(
                composite=DatasetPreprocessorComposite(), leafs=v
            )
            if isinstance(v, list)
            else v
            for k, v in (transforms or {}).items()
        }
        super().__init__(
            dataset_cls=TorchTransformsDataset,
            dataset_kwargs={
                'transforms': _transforms,
                'output_keys': output_keys,
                'sample_as_dict': sample_as_dict,
            },
        )

    @property
    def dataframe_kwarg_name(self) -> str:
        return 'df'

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from pandas import DataFrame


class DFUtils:
    @staticmethod
    def proportion_split_dataframe(
        df: 'DataFrame', proportions: List[float]
    ) -> List['DataFrame']:
        assert sum(proportions) == 1, 'Proportions must sum to 1'
        _df = df.sample(frac=1).reset_index().rename(columns={'index': '_src_index'})
        split_sizes = (np.array(proportions) * len(df)).astype(int)
        split_sizes[-1] += len(df) - split_sizes.sum()  # Adjust for rounding errors

        split_indices = np.cumsum(split_sizes)

        return [
            df.iloc[start:end]
            for start, end in zip(
                [0] + list(split_indices[:-1]), split_indices, strict=True
            )
        ]

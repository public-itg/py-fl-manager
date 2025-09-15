from typing import Optional

import pandas as pd

from fl_manager.core.schemas.dataset import GenericDataset


class PandasDataset(GenericDataset):
    train: pd.DataFrame
    val: Optional[pd.DataFrame] = None
    test: Optional[pd.DataFrame] = None

    def get_full_dataset(self):
        val = self.val if self.val is not None else pd.DataFrame()
        test = self.test if self.test is not None else pd.DataFrame()
        return pd.concat([self.train, val, test])

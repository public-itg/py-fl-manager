import pandera.pandas as pa

from fl_manager.core.components.validators.base import BasePanderaDatasetValidator


def test_pandera_dataset_validator(mocker, pandas_dataset_alpha):
    _dataframe_schema = pa.DataFrameSchema(
        columns={
            'even_feature': pa.Column(
                checks=pa.Check(check_fn=lambda s: all([e % 2 == 0 for e in s]))
            ),
            'odd_feature': pa.Column(
                checks=pa.Check(check_fn=lambda s: all([e % 2 != 0 for e in s]))
            ),
        }
    )
    validator = BasePanderaDatasetValidator(columns={})
    mocker.patch.object(validator, '_dataframe_schema', new=_dataframe_schema)
    validator.validate(pandas_dataset_alpha)

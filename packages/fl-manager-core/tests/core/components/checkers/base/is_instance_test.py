import pandera.pandas as pa

from fl_manager.core.components.checkers.base import IsInstance


def test_is_instance_checker(pandas_dataset_alpha):
    checker = IsInstance(instance_cls=['int'])
    schema = pa.DataFrameSchema(
        {
            'even_feature': pa.Column(int, [checker.get_checker()]),
            'odd_feature': pa.Column(int, [checker.get_checker()]),
        }
    )
    schema.validate(pandas_dataset_alpha.train)

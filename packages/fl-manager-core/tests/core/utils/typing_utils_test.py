from typing import Optional, Any, get_args

from fl_manager.core.utils.typing_utils import TypingUtils


def test_fixed_class():
    class _A:
        def __init__(self, a: int, b: float):
            self._a = a
            self._b = b

    _schema = TypingUtils.get_class_init_args_as_schema(_A)
    assert _schema.model_config.get('extra', None) == 'ignore', 'expected "ignore"'
    assert all([k in _schema.model_fields for k in ['a', 'b']]), 'missing fields'
    assert _schema.model_fields['a'].annotation is int
    assert _schema.model_fields['a'].is_required()
    assert _schema.model_fields['b'].annotation is float
    assert _schema.model_fields['b'].is_required()


def test_optional_class():
    class _A:
        def __init__(self, b: Optional[float] = None):
            self._b = b

    _schema = TypingUtils.get_class_init_args_as_schema(_A)
    assert float in get_args(_schema.model_fields['b'].annotation)
    assert not _schema.model_fields['b'].is_required()


def test_extra_class():
    class _A:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

    _schema = TypingUtils.get_class_init_args_as_schema(_A)
    assert _schema.model_config.get('extra', None) == 'allow', 'expected "allow"'


def test_arbitrary_class():
    from fl_manager.core.meta_registry import MetaRegistry

    class _A:
        def __init__(self, a: MetaRegistry):
            self._a = a

    _schema = TypingUtils.get_class_init_args_as_schema(_A)
    assert _schema.model_fields['a'].annotation == Any

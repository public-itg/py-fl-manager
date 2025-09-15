import collections
import inspect
import types
from typing import Any, get_origin, get_args, Dict

from pydantic import BaseModel, TypeAdapter, PydanticUserError, create_model


class TypingUtils:
    _schema_cache: Dict[type, type[BaseModel]] = {}

    @classmethod
    def get_class_init_args_as_schema(cls, in_class: type) -> type[BaseModel]:
        """
        Generated a pydantic BaseModel according to the class init arguments.
        """
        if cls._schema_cache.get(in_class) is None:
            in_params = inspect.signature(in_class.__init__).parameters
            cls_kwargs = {'extra': 'ignore'}
            params = {}
            for k, v in in_params.items():
                if v.kind in [
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ]:
                    cls_kwargs['extra'] = 'allow'
                    continue
                if v.name == 'self':
                    continue
                annotation = (
                    Any if v.annotation == inspect.Parameter.empty else v.annotation
                )
                annotation = cls._type_simplification(annotation)
                default = ... if v.default == inspect.Parameter.empty else v.default
                params[k] = (annotation, default)
            cls._schema_cache[in_class] = create_model(
                f'{in_class.__name__}Schema', __cls_kwargs__=cls_kwargs, **params
            )
        return cls._schema_cache[in_class]

    @staticmethod
    def _type_simplification(tp: Any) -> Any:
        """
        Recursively replace non-basic and non-pydantic-compatible types with Any.
        """
        origin = get_origin(tp)
        args = get_args(tp)

        if origin is None:  # If it's not a generic type
            try:
                TypeAdapter(tp).json_schema()
                if issubclass(tp, BaseModel) and tp.model_config.get(
                    'arbitrary_types_allowed', False
                ):
                    return Any
                return tp
            except (PydanticUserError, Exception):
                return Any
        elif origin is type:
            return Any
        elif inspect.isclass(origin) and issubclass(origin, collections.abc.Callable):
            return Any
        elif origin is types.UnionType:  # fixme, maybe will fail
            return Any

        simplified_args = tuple(TypingUtils._type_simplification(arg) for arg in args)
        return origin[simplified_args]  # noqa

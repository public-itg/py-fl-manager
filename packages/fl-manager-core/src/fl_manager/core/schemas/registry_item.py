import logging
from functools import cached_property
from typing import Dict, Any, Optional

from pydantic import BaseModel, model_validator, field_validator

from fl_manager.core.meta_registry import MetaRegistry
from fl_manager.core.utils.typing_utils import TypingUtils

logger = logging.getLogger(__name__)


class RegistryItem(BaseModel):
    """
    A schema that contains information about a registry item.

    Attributes:
        registry_id (str): Registry ID.
        name (str): Name of registered item in the registry.
        keyword_arguments Optional(Dict[str, Any]): Keyword arguments for item instantiation.
    """

    registry_id: str
    name: str
    keyword_arguments: Optional[Dict[str, Any]] = None

    @field_validator('registry_id')
    def check_registry_id(cls, v: str) -> str:
        """
        Checks if a Registry exists with the given id.

        Args:
            v (str): Registry id.

        Returns:
            str: The registry id.
        """
        MetaRegistry.get(v)
        return v

    @model_validator(mode='after')
    def verify_registry_item(self) -> 'RegistryItem':
        """
        Check if given name exists in the registry.

        Returns:
            RegistryItem: The registry item.
        """
        _reg = MetaRegistry.get(self.registry_id)
        _reg.get(self.name)
        return self

    @model_validator(mode='after')
    def verify_keyword_arguments(self) -> 'RegistryItem':
        """
        Validator that replaces dict occurrences for actual RegistryItem if suitable.

        Returns:
            RegistryItem: The same instance after validation.
        """
        if self.keyword_arguments is None:
            self.keyword_arguments = {}
        self.keyword_arguments = {
            k: self._validate_keyword_arguments(v)
            for k, v in self.keyword_arguments.items()
        }
        TypingUtils.get_class_init_args_as_schema(
            MetaRegistry.get(self.registry_id).get(self.name)
        ).model_validate(self.keyword_arguments)
        return self

    @cached_property
    def instance(self) -> Any:
        """
        Create an instance with the object configuration.

        Returns:
            Any: Instance of the registry item.
        """
        keyword_args = {
            k: self._parse_value(v) for k, v in (self.keyword_arguments or {}).items()
        }
        logger.info(
            f'Creating ({self.name}) from ({self.registry_id}) with: {keyword_args}.'
        )
        return MetaRegistry.get(self.registry_id).create(self.name, **keyword_args)

    def _validate_keyword_arguments(self, value: Any) -> Any:
        """
        Iterates thought input and converts recursively dictionaries that are other `RegistryItem`.

        Parameters:
            value: Any

        Returns:
            Any: Same as input but with suitable dict converted to :py:class:`RegistryItem`.
        """
        if isinstance(value, dict):
            if 'registry_id' in value:
                return RegistryItem.model_validate(value)
            return {k: self._validate_keyword_arguments(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._validate_keyword_arguments(v) for v in value]
        return value

    def _parse_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._parse_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._parse_value(v) for v in value]
        if isinstance(value, RegistryItem):
            return value.instance
        return value

import json
import re
from typing import Optional, Any, List

from pydantic import BaseModel, model_validator


class BaseVariables(BaseModel):
    """
    Base class for schemas that allow variables marked with '{{NAME}}'.
    """

    variables: Optional[dict[str, Any]] = None

    @model_validator(mode='before')
    @classmethod
    def parse_variables(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return cls._extract_and_substitute_variables(data)
        return data

    @classmethod
    def _extract_and_substitute_variables(cls, data: dict) -> dict:
        _variables = data.pop('variables', None)
        _variables = _variables or {}
        if not isinstance(_variables, dict):
            raise ValueError('Invalid variables type.')
        _vars = cls._extract_variables(json.dumps(data))
        if _unset_vars := set(_vars).difference(set(_variables.keys())):
            raise ValueError(f'missing variables: {_unset_vars}')
        _data = cls._substitute_variables(_variables, data)
        assert isinstance(_data, dict), (
            'something went wrong during components substitution.'
        )
        _data['variables'] = _variables
        return _data

    @classmethod
    def _extract_variables(cls, data: str) -> List[str]:
        # Regular expression to match text inside double angle brackets
        pattern = r'\{\{(.*?)\}\}'
        # Find all matches in the input string
        matches = re.findall(pattern, data)
        return matches

    @classmethod
    def _substitute_variables(cls, params: dict, structure: dict | list | str) -> Any:
        if isinstance(structure, dict):
            return {
                k: cls._substitute_variables(params, v) for k, v in structure.items()
            }
        elif isinstance(structure, list):
            return [cls._substitute_variables(params, item) for item in structure]
        elif isinstance(structure, str):
            if structure.startswith('{{') and structure.endswith('}}'):
                param_name = structure[2:-2]
                return params.get(
                    param_name, structure
                )  # Return the placeholder if param not found
            return structure
        else:
            return structure

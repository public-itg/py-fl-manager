from pydantic import BaseModel

from fl_manager.core.utils.typing_utils import TypingUtils


class Component:
    @classmethod
    def get_schema(cls) -> type[BaseModel]:
        """
        Defines the schema of the input variables of this component.
        """
        return TypingUtils.get_class_init_args_as_schema(cls)

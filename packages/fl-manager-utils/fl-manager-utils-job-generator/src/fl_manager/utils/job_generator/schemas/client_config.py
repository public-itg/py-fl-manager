from typing import Optional, Dict, Union, List

from pydantic import BaseModel, model_validator

from fl_manager.core.schemas.registry_item import RegistryItem


class ClientConfig(BaseModel):
    components: Optional[Dict[str, Union[RegistryItem, List[RegistryItem]]]] = None

    @model_validator(mode='after')
    def verify_keyword_arguments(self):
        if self.components is None:
            self.components = {}
        self.components = {
            k: self._validate_components(v) for k, v in self.components.items()
        }
        return self

    @staticmethod
    def _validate_components(
        component: RegistryItem | List[RegistryItem],
    ) -> RegistryItem | List[RegistryItem]:
        if isinstance(component, RegistryItem):
            component = [component]
        assert all(
            c.registry_id in ['dataset_reader', 'dataset_formatter'] for c in component
        ), 'client can only customize "dataset_reader" and "dataset_formatter"'
        return component

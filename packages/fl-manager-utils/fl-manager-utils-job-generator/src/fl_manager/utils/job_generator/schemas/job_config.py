from typing import Optional, Dict, Union, List

from pydantic import BaseModel, model_validator

from fl_manager.core.schemas.registry_item import RegistryItem
from fl_manager.utils.job_generator.schemas.client_config import ClientConfig


class FLAlgorithmConfig(BaseModel):
    name: str
    server_kwargs: Optional[dict] = None
    client_kwargs: Optional[dict] = None

    @model_validator(mode='after')
    def prefill_optional_dicts(self):
        if self.server_kwargs is None:
            self.server_kwargs = {}
        if self.client_kwargs is None:
            self.client_kwargs = {}
        return self


class JobConfig(BaseModel):
    name: str = 'federated_job'
    num_rounds: int
    executor: str
    executor_kwargs: Optional[dict] = None
    runner: str
    fl_algorithm: FLAlgorithmConfig
    components: Dict[str, Union[RegistryItem, List[RegistryItem]]]
    clients: Dict[str, ClientConfig]

    @model_validator(mode='after')
    def prefill_optional_dicts(self):
        if self.executor_kwargs is None:
            self.executor_kwargs = {}
        return self

    @model_validator(mode='after')
    def check_clients(self):
        assert len(self.clients) != 0
        return self

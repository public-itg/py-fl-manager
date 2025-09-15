from typing import Optional, List, Union, Dict, cast, Any

from nvflare.apis.dxo import DataKind
from nvflare.apis.impl.controller import Controller
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_opt.pt import PTFedOptModelShareableGenerator
from torch import nn

from fl_manager.utils.job_generator.federated_job.pt.fed_job.base_sag_fed_job import (
    BaseSAGFedJob,
)


class PTFedOptModelShareableGeneratorWrapper(PTFedOptModelShareableGenerator):
    def __init__(
        self,
        source_model: Union[str, nn.Module],
        optimizer_args: Optional[dict] = None,
        lr_scheduler_args: Optional[dict] = None,
    ):
        _kwargs = {}
        if optimizer_args is not None:
            _kwargs['optimizer_args'] = optimizer_args
        if lr_scheduler_args is not None:
            _kwargs['lr_scheduler_args'] = lr_scheduler_args
        super().__init__(source_model=cast(Any, source_model), **_kwargs)


class FedOptJob(BaseSAGFedJob):
    def __init__(
        self,
        initial_model: nn.Module,
        n_clients: int,
        num_rounds: int,
        name: str = 'fed_job',
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = 'accuracy',
        negate_key_metric: bool = False,
        model_persistor: Optional[ModelPersistor] = None,
        optimizer_args: Optional[dict] = None,
        lr_scheduler_args: Optional[dict] = None,
        pre_train_controllers: Optional[Dict[str, Controller]] = None,
        post_train_controllers: Optional[Dict[str, Controller]] = None,
    ):
        # Source model can be instance or id, if id, the persistor should be changed too
        shareable_generator = PTFedOptModelShareableGeneratorWrapper(
            source_model=initial_model,
            optimizer_args=optimizer_args,
            lr_scheduler_args=lr_scheduler_args,
        )
        super().__init__(
            initial_model=initial_model,
            n_clients=n_clients,
            num_rounds=num_rounds,
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            key_metric=key_metric,
            negate_key_metric=negate_key_metric,
            model_persistor=model_persistor,
            expected_data_kind=cast(DataKind, DataKind.WEIGHT_DIFF),
            shareable_generator=shareable_generator,
            pre_train_controllers=pre_train_controllers,
            post_train_controllers=post_train_controllers,
        )

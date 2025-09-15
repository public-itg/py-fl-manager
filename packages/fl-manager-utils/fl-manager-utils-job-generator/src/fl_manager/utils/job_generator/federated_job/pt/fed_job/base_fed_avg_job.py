from typing import Optional, List, Dict

from nvflare.apis.impl.controller import Controller
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from torch import nn


class BaseFedAvgJob(BaseFedJob):
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
        pre_train_controllers: Optional[Dict[str, Controller]] = None,
        post_train_controllers: Optional[Dict[str, Controller]] = None,
    ):
        """PyTorch FedAvg Job.

        Configures server side FedAvg controller, persistor with initial model, and widgets.

        User must add executors.

        Reference: FedAvgJob (fed_avg.py from nvflare)
        """
        if not isinstance(initial_model, nn.Module):
            raise ValueError(
                f'Expected initial model to be nn.Module, but got type f{type(initial_model)}.'
            )

        _intime_model_selector = IntimeModelSelector(
            key_metric=key_metric, negate_key_metric=negate_key_metric
        )

        super().__init__(
            initial_model=initial_model,
            name=name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            key_metric=key_metric,
            intime_model_selector=_intime_model_selector,
            model_persistor=model_persistor,
        )

        for k, v in (pre_train_controllers or {}).items():
            self.to_server(obj=v, id=k)

        self.to_server(
            obj=FedAvg(
                num_clients=n_clients,
                num_rounds=num_rounds,
                persistor_id=self.comp_ids['persistor_id'],
            ),
            id='fed_avg_ctl',
        )

        for k, v in (post_train_controllers or {}).items():
            self.to_server(obj=v, id=k)

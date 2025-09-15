from typing import Optional, List, Dict, cast

from nvflare.apis.dxo import DataKind
from nvflare.apis.impl.controller import Controller
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.api import validate_object_for_job
from torch import nn


class BaseSAGFedJob(BaseFedJob):
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
        expected_data_kind: str | DataKind = DataKind.WEIGHTS,
        shareable_generator: Optional[ShareableGenerator] = None,
        aggregator: Optional[Aggregator] = None,
        pre_train_controllers: Optional[Dict[str, Controller]] = None,
        post_train_controllers: Optional[Dict[str, Controller]] = None,
    ):
        """PyTorch ScatterAndGather Job.

        Configures server side ScatterAndGather controller, persistor with initial model, and widgets.

        User must add executors.

        Reference: SAGMLFlowJob (fed_sag_mlflow.py from nvflare).
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

        if shareable_generator:
            validate_object_for_job(
                'shareable_generator', shareable_generator, ShareableGenerator
            )
        else:
            shareable_generator = FullModelShareableGenerator()
        shareable_generator_id = self.to_server(
            shareable_generator, id='shareable_generator'
        )

        if aggregator:
            validate_object_for_job('aggregator', aggregator, Aggregator)
        else:
            aggregator = InTimeAccumulateWeightedAggregator(
                expected_data_kind=cast(DataKind, expected_data_kind)
            )
        aggregator_id = self.to_server(aggregator, id='aggregator')

        for k, v in (pre_train_controllers or {}).items():
            self.to_server(obj=v, id=k)

        self.to_server(
            obj=ScatterAndGather(
                min_clients=n_clients,
                num_rounds=num_rounds,
                wait_time_after_min_received=10,
                aggregator_id=aggregator_id,
                persistor_id=self.comp_ids['persistor_id'],
                shareable_generator_id=shareable_generator_id,
            ),
            id='sag_ctl',
        )

        for k, v in (post_train_controllers or {}).items():
            self.to_server(obj=v, id=k)

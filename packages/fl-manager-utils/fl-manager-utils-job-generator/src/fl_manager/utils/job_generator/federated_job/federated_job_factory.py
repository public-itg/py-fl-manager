import inspect
from types import ModuleType

from fl_manager.core.components.models import FederatedLearningModel
from fl_manager.utils.job_generator.federated_job.federated_job import FederatedJob


class FederatedJobFactory:
    @classmethod
    def create(
        cls, fl_model: FederatedLearningModel, fl_algorithm: str
    ) -> FederatedJob:
        _bases = set(
            [
                (j.__package__ or j.__name__).partition('.')[0]
                for j in [
                    inspect.getmodule(e) for e in inspect.getmro(fl_model.model_cls)
                ]
                if isinstance(j, ModuleType)
            ]
        )
        if not {'torch', 'lightning'}.isdisjoint(_bases):
            from .pt.pt_federated_job_registry import PTFederatedJobRegistry

            return PTFederatedJobRegistry.create(fl_algorithm, fl_model=fl_model)
        raise RuntimeError(f'Invalid model type. Bases: {_bases}')

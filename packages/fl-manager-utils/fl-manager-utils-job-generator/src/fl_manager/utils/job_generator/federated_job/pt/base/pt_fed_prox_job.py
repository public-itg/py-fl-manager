from fl_manager.utils.job_generator.federated_job.pt.base._pt_base_fed_avg_job import (
    PTBaseFedAvgJob,
)
from fl_manager.utils.job_generator.federated_job.pt.pt_federated_job_registry import (
    PTFederatedJobRegistry,
)


@PTFederatedJobRegistry.register(name='fed_prox')
class PTFedProxJob(PTBaseFedAvgJob):
    pass

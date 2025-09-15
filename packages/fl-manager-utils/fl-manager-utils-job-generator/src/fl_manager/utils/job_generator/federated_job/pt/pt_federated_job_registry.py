from fl_manager.core.utils.registry import ClassRegistry

from fl_manager.utils.job_generator.federated_job.pt.pt_federated_job import (
    PTFederatedJob,
)

PTFederatedJobRegistry = ClassRegistry[PTFederatedJob](PTFederatedJob)

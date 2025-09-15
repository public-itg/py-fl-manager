from fl_manager.core.utils.import_utils import ImportUtils
from fl_manager.utils.job_generator.federated_job.pt.pt_federated_job_registry import (
    PTFederatedJobRegistry,
)

__all__ = ['PTFederatedJobRegistry']

ImportUtils.iter_import_pkg('.pt', 'fl_manager.utils.job_generator.federated_job')

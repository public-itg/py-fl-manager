import logging

from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob

from fl_manager.utils.job_generator.federated_job.pt.base._pt_base_fed_avg_job import (
    PTBaseFedAvgJob,
)
from fl_manager.utils.job_generator.federated_job.pt.pt_federated_job_registry import (
    PTFederatedJobRegistry,
)
from fl_manager.utils.nvflare_extensions.nv.workflows.pfl_model_export_controller import (
    PFLModelExportController,
)

logger = logging.getLogger(__name__)


@PTFederatedJobRegistry.register(name='fed_bn')
class PTFedBNJob(PTBaseFedAvgJob):
    def _add_export_controller(self, job: BaseFedJob):
        job.to_server(PFLModelExportController(), 'model_fit_and_export_controller')

# NVFlare `FedJob` for PyTorch

## Base

::: fl_manager.utils.job_generator.federated_job.pt.fed_job.base_fed_avg_job
    options:
      members:
      - BaseFedAvgJob
      show_source: false

::: fl_manager.utils.job_generator.federated_job.pt.fed_job.base_sag_fed_job
    options:
      members:
      - BaseSAGFedJob
      show_source: false

## FedAvg

::: fl_manager.utils.job_generator.federated_job.pt.fed_job.fed_avg
    options:
      members:
      - FedAvgJob
      show_source: false

## FedOpt

::: fl_manager.utils.job_generator.federated_job.pt.fed_job.fed_opt
    options:
      members:
      - FedOptJob
      - PTFedOptModelShareableGeneratorWrapper
      show_source: false

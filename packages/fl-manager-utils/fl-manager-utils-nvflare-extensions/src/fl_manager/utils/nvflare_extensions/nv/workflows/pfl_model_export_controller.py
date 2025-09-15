from nvflare.apis.controller_spec import Task, ClientTask
from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants

from fl_manager.utils.nvflare_extensions.nv.workflows.base_controller_with_client_selection import (
    BaseControllerWithClientSelection,
)


class PFLModelExportController(BaseControllerWithClientSelection):
    def __init__(
        self,
        fit_and_export_model_task_name: str = 'fit_and_export_model',
        fit_and_export_model_task_timeout: int = 0,
        participating_clients: list[str] | None = None,
        wait_for_clients_timeout: int = 300,
    ):
        super().__init__(
            participating_clients=participating_clients,
            wait_for_clients_timeout=wait_for_clients_timeout,
        )
        self._fit_and_export_model_task_name = fit_and_export_model_task_name
        self._fit_and_export_model_task_timeout = fit_and_export_model_task_timeout

    def setup_controller(self, fl_ctx: FLContext):
        """No additional setups needed"""
        pass

    def run_control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        success = self._fit_and_export_global_model(abort_signal, fl_ctx)
        if not success:
            self.log_info(fl_ctx, 'Could not fit and export global model.')

    def _fit_and_export_global_model(
        self, abort_signal: Signal, fl_ctx: FLContext
    ) -> bool:
        _task_name = 'fit_and_export_global_model'
        self._send_model_fit_and_export_task(fl_ctx)
        return self.wait_until_task_complete(
            task_name=_task_name, abort_signal=abort_signal, fl_ctx=fl_ctx
        )

    def _send_model_fit_and_export_task(self, fl_ctx: FLContext):
        self.log_info(
            fl_ctx,
            'Sending last model to all participating clients for fit and export.',
        )
        task = Task(
            name=self._fit_and_export_model_task_name,
            data=Shareable(),
            before_task_sent_cb=self._before_task_sent_cb,
            after_task_sent_cb=self._after_task_sent_cb,
            timeout=self._fit_and_export_model_task_timeout,
        )
        self.broadcast(
            task=task,
            fl_ctx=fl_ctx,
            targets=self._participating_clients,
            min_responses=len(self._participating_clients),
            wait_time_after_min_received=0,
        )

    def _before_task_sent_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        model_dxo: DXO = model_learnable_to_dxo(
            fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        )
        model_shareable = model_dxo.to_shareable()
        client_task.task.data = model_shareable

        fl_ctx.set_prop(
            AppConstants.DATA_CLIENT, client_task.client, private=True, sticky=False
        )
        fl_ctx.set_prop(
            '_model_to_fit_and_export_', model_shareable, private=True, sticky=False
        )
        fl_ctx.set_prop(
            AppConstants.PARTICIPATING_CLIENTS,
            self._participating_clients,
            private=True,
            sticky=False,
        )
        self.fire_event('_send_model_for_fit_and_export', fl_ctx)

    def _after_task_sent_cb(self, client_task: ClientTask, fl_ctx: FLContext):  # noqa
        # Once task is sent clear data to restore memory
        client_task.task.data = None

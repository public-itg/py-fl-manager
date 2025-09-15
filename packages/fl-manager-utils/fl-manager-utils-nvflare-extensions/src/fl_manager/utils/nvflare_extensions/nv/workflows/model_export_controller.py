from collections import defaultdict
from typing import Dict, Any

from nvflare.apis.controller_spec import Task, ClientTask
from nvflare.apis.dxo import DXO, from_shareable, DataKind, get_leaf_dxos
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import (
    make_model_learnable,
    validate_model_learnable,
)
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.security.logging import secure_format_exception

from fl_manager.utils.nvflare_extensions.nv.workflows.base_controller_with_client_selection import (
    BaseControllerWithClientSelection,
)


class ModelExportController(BaseControllerWithClientSelection):
    """
    Based on `CrossSiteModelEval` controller and `IntTimeModelSelector` from `nvflare`.
    """

    def __init__(
        self,
        model_locator_id: str,
        val_dir: str = 'model_export_val',
        validation_task_name: str = AppConstants.TASK_VALIDATION,
        validation_task_timeout: int = 6000,
        export_model_task_name: str = 'export_model',
        export_model_task_timeout: int = 600,
        participating_clients: list[str] | None = None,
        wait_for_clients_timeout: int = 300,
        key_metric: str = 'val_accuracy',
        negate_key_metric: bool = False,
    ):
        super().__init__(
            participating_clients=participating_clients,
            wait_for_clients_timeout=wait_for_clients_timeout,
        )
        self._model_locator_id = model_locator_id
        self._val_dir = val_dir
        self._validation_task_name = validation_task_name
        self._validation_task_timeout = validation_task_timeout
        self._export_model_task_name = export_model_task_name
        self._export_model_task_timeout = export_model_task_timeout
        self._key_metric = key_metric
        self._negate_key_metric = negate_key_metric
        self._server_models = {}
        self._val_results = {}
        self._model_locator: ModelLocator | None = None

    def setup_controller(self, fl_ctx: FLContext):
        self._model_locator = self.get_component(
            component_id=self._model_locator_id,
            expected_cls=ModelLocator,
            fl_ctx=fl_ctx,
        )

    def run_control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        success_validate_server_models = self._validate_server_models(
            abort_signal, fl_ctx
        )
        if not success_validate_server_models:
            self.log_info(
                fl_ctx, 'Could not perform server models validation on clients.'
            )
            return
        success_export_best_global_model = self._export_best_global_model(
            abort_signal, fl_ctx
        )
        if not success_export_best_global_model:
            self.log_info(fl_ctx, 'Could not export best global model.')

    def _validate_server_models(self, abort_signal: Signal, fl_ctx: FLContext) -> bool:
        _task_name = 'validate_server_models'
        success = self._locate_server_models(fl_ctx)
        if not success:
            self.log_info(
                fl_ctx, f'Could not retrieve server models. Finishing {_task_name}.'
            )
            return False

        for server_model in self._server_models.keys():
            self._send_model_validate_task(server_model, fl_ctx)

        return self.wait_until_task_complete(
            task_name=_task_name, abort_signal=abort_signal, fl_ctx=fl_ctx
        )

    def _export_best_global_model(
        self, abort_signal: Signal, fl_ctx: FLContext
    ) -> bool:
        _task_name = 'export_best_global_model'
        _results = self._process_val_results()
        _best_model_name = max(_results, key=_results.get)
        _best_val_metric = _results.get(_best_model_name)
        self._send_model_export_task(_best_model_name, fl_ctx)
        # Refresh server best model
        _best_model: DXO = self._server_models[_best_model_name]
        _model_learnable = make_model_learnable(_best_model.data, _best_model.meta)
        validate_model_learnable(_model_learnable)
        self.log_info(fl_ctx, f'Refresh best model to {_best_model_name}.')
        fl_ctx.set_prop(
            AppConstants.GLOBAL_MODEL, _model_learnable, private=True, sticky=True
        )
        fl_ctx.set_prop(
            AppConstants.VALIDATION_RESULT, _best_val_metric, private=True, sticky=False
        )
        self.fire_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)

        return self.wait_until_task_complete(
            task_name=_task_name, abort_signal=abort_signal, fl_ctx=fl_ctx
        )

    def _locate_server_models(self, fl_ctx: FLContext) -> bool:
        # Load models from model_locator
        self.log_info(fl_ctx, 'Locating server models.')
        server_model_names = self._model_locator.get_model_names(fl_ctx)

        unique_names = []
        for name in server_model_names:
            dxo = self._model_locator.locate_model(name, fl_ctx)
            if not isinstance(dxo, DXO):
                self.system_panic(
                    f'ModelLocator produced invalid data: expect DXO but got {type(dxo)}.',
                    fl_ctx,
                )
                return False
            unique_name = f'SRV_{name}'
            unique_names.append(unique_name)
            self._server_models[unique_name] = dxo

        if unique_names:
            self.log_info(fl_ctx, f'Server model loaded: {len(unique_names)}.')
        else:
            self.log_info(fl_ctx, 'no server models to export!')
        return True

    def _send_model_validate_task(self, model_name: str, fl_ctx: FLContext):
        self.log_info(
            fl_ctx, f'Sending {model_name} to all participating clients for validation.'
        )
        task = Task(
            name=self._validation_task_name,
            data=Shareable(),
            before_task_sent_cb=self._before_send_export_task_cb,
            after_task_sent_cb=self._after_send_export_task_cb,
            result_received_cb=self._receive_val_result_cb,
            timeout=self._validation_task_timeout,
            props={AppConstants.MODEL_OWNER: model_name},
        )
        self.broadcast(
            task=task,
            fl_ctx=fl_ctx,
            targets=self._participating_clients,
            min_responses=len(self._participating_clients),
            wait_time_after_min_received=0,
        )

    def _send_model_export_task(self, model_name: str, fl_ctx: FLContext):
        self.log_info(
            fl_ctx, f'Sending {model_name} to all participating clients for export.'
        )
        task = Task(
            name=self._export_model_task_name,
            data=Shareable(),
            before_task_sent_cb=self._before_send_export_task_cb,
            after_task_sent_cb=self._after_send_export_task_cb,
            timeout=self._validation_task_timeout,
            props={AppConstants.MODEL_OWNER: model_name},
        )
        self.broadcast(
            task=task,
            fl_ctx=fl_ctx,
            targets=self._participating_clients,
            min_responses=len(self._participating_clients),
            wait_time_after_min_received=0,
        )

    def _before_send_export_task_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        model_name = client_task.task.props[AppConstants.MODEL_OWNER]
        model_dxo: DXO = self._server_models.get(model_name)
        model_dxo.update_meta_props({'model_name': model_name})
        model_shareable = model_dxo.to_shareable()
        model_shareable.set_header(AppConstants.MODEL_OWNER, model_name)
        model_shareable.add_cookie(AppConstants.MODEL_OWNER, model_name)
        client_task.task.data = model_shareable

        fl_ctx.set_prop(
            AppConstants.DATA_CLIENT, client_task.client, private=True, sticky=False
        )
        fl_ctx.set_prop(
            AppConstants.MODEL_OWNER, model_name, private=True, sticky=False
        )
        fl_ctx.set_prop(
            '_model_to_export_', model_shareable, private=True, sticky=False
        )
        fl_ctx.set_prop(
            AppConstants.PARTICIPATING_CLIENTS,
            self._participating_clients,
            private=True,
            sticky=False,
        )
        self.fire_event('_send_model_for_export', fl_ctx)

    def _after_send_export_task_cb(self, client_task: ClientTask, fl_ctx: FLContext):  # noqa
        # Once task is sent clear data to restore memory
        client_task.task.data = None

    def _receive_val_result_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        # Find name of the client sending this
        result = client_task.result
        client_name = client_task.client.name

        self._accept_val_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

        client_task.result = None

    def _accept_val_result(
        self, client_name: str, result: Shareable, fl_ctx: FLContext
    ):
        model_owner = result.get_cookie(AppConstants.MODEL_OWNER, '')

        # Fire event. This needs to be a new local context per each client
        fl_ctx.set_prop(
            AppConstants.MODEL_OWNER, model_owner, private=True, sticky=False
        )
        fl_ctx.set_prop(
            AppConstants.DATA_CLIENT, client_name, private=True, sticky=False
        )
        fl_ctx.set_prop(
            AppConstants.VALIDATION_RESULT, result, private=True, sticky=False
        )
        self.fire_event(AppEventType.VALIDATION_RESULT_RECEIVED, fl_ctx)

        rc = result.get_return_code()
        if rc and rc != ReturnCode.OK:
            self._process_invalid_accept_val_rc(rc, fl_ctx, client_name, model_owner)
        else:
            if client_name not in self._val_results:
                self._val_results[client_name] = {}
            self._val_results[client_name][model_owner] = {}
            try:
                self._get_val_results(result, fl_ctx, client_name, model_owner)
            except ValueError as e:
                reason = (
                    f'Bad validation result from {client_name} on model {model_owner}. '
                    f'Exception: {secure_format_exception(e)}'
                )
                self.log_exception(fl_ctx, reason)
            except Exception as e:
                reason = (
                    f'Exception in handling validation result. '
                    f'Exception: {secure_format_exception(e)}'
                )
                self.log_exception(fl_ctx, reason)

    def _get_val_results(
        self, result: Shareable, fl_ctx: FLContext, client_name: str, model_owner: str
    ):
        dxo = from_shareable(result)
        dxo.validate()
        if dxo.data_kind == DataKind.METRICS:
            if client_name not in self._val_results:
                self._val_results[client_name] = {}
            self._val_results[client_name][model_owner] = dxo.data
        elif dxo.data_kind == DataKind.COLLECTION:
            # The DXO could contain multiple sub-DXOs (e.g. received from a T2 system)
            leaf_dxo, errors = get_leaf_dxos(dxo, client_name)
            if errors:
                for err in errors:
                    self.log_error(fl_ctx, f'Bad result from {client_name}: {err}')
            for _sub_data_client, _dxo in leaf_dxo.items():
                _dxo.validate()
                if _sub_data_client not in self._val_results:
                    self._val_results[_sub_data_client] = {}
                self._val_results[_sub_data_client][model_owner] = _dxo.data
        else:
            self.log_error(
                fl_ctx,
                f'Expected dxo of kind METRICS or COLLECTION but got {dxo.data_kind} instead.',
                fire_event=False,
            )

    def _process_invalid_accept_val_rc(
        self, rc: Any, fl_ctx: FLContext, client_name: str, model_owner: str
    ):
        # Raise errors if bad peer context or execution exception.
        if rc in [ReturnCode.MISSING_PEER_CONTEXT, ReturnCode.BAD_PEER_CONTEXT]:
            self.log_error(fl_ctx, 'Peer context is bad or missing.')
        elif rc in [ReturnCode.EXECUTION_EXCEPTION, ReturnCode.TASK_UNKNOWN]:
            self.log_error(fl_ctx, 'Execution Exception in model validation.')
        elif rc in [
            ReturnCode.EXECUTION_RESULT_ERROR,
            ReturnCode.TASK_DATA_FILTER_ERROR,
            ReturnCode.TASK_RESULT_FILTER_ERROR,
        ]:
            self.log_error(
                fl_ctx,
                'Execution result is not a shareable. Validation results will be ignored.',
            )
        else:
            self.log_error(
                fl_ctx,
                f'Client {client_name} sent results for validating {model_owner} model with return code set.'
                ' Logging empty results.',
            )

        if client_name not in self._val_results:
            self._val_results[client_name] = {}
        self._val_results[client_name][model_owner] = {}

    def _process_val_results(self) -> Dict[str, float]:
        sums = defaultdict(float)
        counts = defaultdict(int)
        _val_results_data = [
            {k: v.get(self._key_metric, 0) for k, v in _v.items()}
            for _v in self._val_results.values()
        ]
        for d in _val_results_data:
            for key, value in d.items():
                sums[key] += (-1.0 * value) if self._negate_key_metric else value
                counts[key] += 1
        _averages = {key: sums[key] / counts[key] for key in sums}
        return _averages

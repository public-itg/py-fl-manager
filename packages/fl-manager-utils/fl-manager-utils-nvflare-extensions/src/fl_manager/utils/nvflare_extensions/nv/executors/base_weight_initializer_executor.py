import abc

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants


class BaseWeightInitializerExecutor(Executor, metaclass=abc.ABCMeta):
    def __init__(self, pre_train_task_name: str = AppConstants.TASK_GET_WEIGHTS):
        super().__init__()
        self._pre_train_task_name = pre_train_task_name

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        try:
            match task_name:
                case self._pre_train_task_name:
                    return self._run_pre_train_task()
                case _:
                    return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f'Exception in simple trainer: {e}.')
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _run_pre_train_task(self) -> Shareable:
        weights = self._get_model_weights()

        outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)
        return outgoing_dxo.to_shareable()

    @abc.abstractmethod
    def _get_model_weights(self) -> dict:
        raise NotImplementedError()

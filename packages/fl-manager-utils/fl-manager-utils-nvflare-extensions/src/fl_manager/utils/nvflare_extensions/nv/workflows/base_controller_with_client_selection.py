import abc
import time

from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal

from fl_manager.utils.nvflare_extensions.nv.workflows.base_controller import (
    BaseController,
)


class BaseControllerWithClientSelection(BaseController, metaclass=abc.ABCMeta):
    def __init__(
        self,
        task_check_period: float = 0.2,
        participating_clients: list[str] | None = None,
        wait_for_clients_timeout: int = 300,
    ):
        super().__init__(task_check_period=task_check_period)
        self._participating_clients = participating_clients
        self._wait_for_clients_timeout = wait_for_clients_timeout

    def _on_before_control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        return self._wait_for_clients(abort_signal, fl_ctx)

    def _wait_for_clients(self, abort_signal: Signal, fl_ctx: FLContext) -> bool:
        if not self._participating_clients:
            self.log_info(
                fl_ctx,
                f'No specified clients, using all available clients for {self.__class__.__name__}.',
            )
        start_time = time.time()
        while not self._participating_clients:
            self._participating_clients = [c.name for c in self._engine.get_clients()]
            if time.time() - start_time > self._wait_for_clients_timeout:
                self.log_info(
                    fl_ctx, f'No clients available - quit {self.__class__.__name__}.'
                )
                return False

            self.log_info(fl_ctx, 'No clients available - waiting ...')
            time.sleep(2.0)
            if abort_signal.triggered:
                self.log_info(
                    fl_ctx,
                    f'Abort signal triggered. Finishing {self.__class__.__name__}.',
                )
                return False
        self.log_info(
            fl_ctx,
            f'Beginning {self.__class__.__name__} with clients {self._participating_clients}.',
        )
        return True

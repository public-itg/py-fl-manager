import abc
import time
from typing import Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.signal import Signal
from nvflare.security.logging import secure_format_exception


class BaseController(Controller, metaclass=abc.ABCMeta):
    """
    Custom Controller with callbacks and utility methods extracted into reusable functions.
    """

    def __init__(self, task_check_period: float = 0.2):
        super().__init__(task_check_period=task_check_period)

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, f'Initializing controller {self.__class__.__name__}')
        self._on_before_setup_controller(fl_ctx)
        self.setup_controller(fl_ctx)
        self._on_after_setup_controller(fl_ctx)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:
            if not self._on_before_control_flow(abort_signal, fl_ctx):
                return
            self.run_control_flow(abort_signal, fl_ctx)
            self._on_after_control_flow(abort_signal, fl_ctx)
        except Exception as e:
            error_msg = f'Exception in {self.__class__.__name__} control_flow: {secure_format_exception(e)}'
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx=fl_ctx)

    def stop_controller(self, fl_ctx: FLContext):
        self._on_before_cancel_all_tasks(fl_ctx)
        self.cancel_all_tasks(fl_ctx=fl_ctx)
        self._on_after_cancel_all_tasks(fl_ctx)

    def get_component(
        self, component_id: str, expected_cls: type[FLComponent], fl_ctx: FLContext
    ) -> Optional[FLComponent]:
        _component = self._engine.get_component(component_id)
        if not isinstance(_component, expected_cls):
            self.system_panic(
                reason=f'bad component {component_id}: expected {expected_cls} but got {type(_component)}',
                fl_ctx=fl_ctx,
            )
            return
        return _component

    def wait_until_task_complete(
        self, task_name: str, abort_signal: Signal, fl_ctx: FLContext
    ) -> bool:
        while self.get_num_standing_tasks():
            if abort_signal.triggered:
                self.log_info(
                    fl_ctx, f'Aborting signal triggered. Finishing {task_name}.'
                )
                return False
            self.log_debug(
                fl_ctx, f'Checking standing tasks to see if {task_name} finished.'
            )
            time.sleep(self._task_check_period)
        return True

    def _on_before_setup_controller(self, fl_ctx: FLContext):
        """Hook that is called before setup controller."""
        pass

    def _on_after_setup_controller(self, fl_ctx: FLContext):
        """Hook that is called after setup controller."""
        pass

    def _on_before_control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> bool:
        """Hook that is called before running control flow. If fails, exit execution."""
        pass

    def _on_after_control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        """Hook that is called after running control flow."""
        pass

    def _on_before_cancel_all_tasks(self, fl_ctx: FLContext):
        """Hook that is called before cancel all tasks."""
        pass

    def _on_after_cancel_all_tasks(self, fl_ctx: FLContext):
        """Hook that is called after cancel all tasks."""
        pass

    @abc.abstractmethod
    def setup_controller(self, fl_ctx: FLContext):
        raise NotImplementedError()

    @abc.abstractmethod
    def run_control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        raise NotImplementedError()

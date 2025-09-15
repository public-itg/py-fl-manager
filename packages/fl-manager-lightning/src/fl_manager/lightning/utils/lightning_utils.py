import logging
from typing import List, Type, Optional, Dict

import torch
from lightning import LightningModule
from torch import Tensor

logger = logging.getLogger(__name__)


class LightningUtils:
    @staticmethod
    def get_metrics(metrics: Dict[str, Tensor]):
        return {k: v.item() for k, v in metrics.items()}

    @staticmethod
    def load_state_dict_skipping_modules(
        pl_module: LightningModule,
        state_dict: dict,
        strict: bool,
        modules_to_skip: Optional[List[Type[object]]] = None,
    ):
        _mts = modules_to_skip if modules_to_skip is not None else []
        _mts = tuple(_mts)
        _new_state_dict = state_dict.copy()
        if len(_mts) != 0:
            mapping = LightningUtils._get_state_dict_module_mapping(
                pl_module, state_dict
            )
            _new_state_dict = {
                k: v
                for k, v in state_dict.items()
                if not issubclass(mapping.get(k, type), _mts)
            }
            logger.info(f'Loading {_new_state_dict.keys()}')
        pl_module.load_state_dict(_new_state_dict, strict=strict)

    @staticmethod
    def get_different_keys(state_dict_a: dict, state_dict_b: dict) -> List[str]:
        different_keys = []
        # Collect keys that are only in one of the dicts
        keys_a = set(state_dict_a.keys())
        keys_b = set(state_dict_b.keys())
        missing_in_dict_b = keys_a - keys_b
        missing_in_dict_a = keys_b - keys_a
        for key in missing_in_dict_b:
            print(f"Key '{key}' missing in second dict")
            different_keys.append(key)
        for key in missing_in_dict_a:
            print(f"Key '{key}' missing in first dict")
            different_keys.append(key)
        # Compare common keys
        common_keys = keys_a & keys_b
        for key in common_keys:
            if not torch.equal(state_dict_a[key], state_dict_b[key]):
                print(f"Difference in values for key '{key}'")
                different_keys.append(key)
        return different_keys

    @staticmethod
    def _get_state_dict_module_mapping(
        pl_module: LightningModule, state_dict: dict
    ) -> dict:
        _sd_keys = state_dict.keys()
        _m_mapping = {}
        for name, module in pl_module.named_modules():
            param_names = [
                param_name for param_name, _ in module.named_parameters(recurse=False)
            ]
            buffer_names = [
                buffer_name for buffer_name, _ in module.named_buffers(recurse=False)
            ]
            for _c_name in param_names + buffer_names:
                full_name = f'{name}.{_c_name}' if name else _c_name
                if full_name in _sd_keys:
                    _m_mapping[full_name] = module.__class__
        return _m_mapping

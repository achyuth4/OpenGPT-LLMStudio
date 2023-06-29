import logging
from copy import copy
from typing import Any, List

import torch
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from app_utils.utils import copy_config
from llm_studio.python_configs.text_causal_language_modeling_config import (
    MODELNAME2MODELTYPE,
    ConfigProblemBase,
)

__all__ = ["ConfigUpdaterFactory"]

from llm_studio.src.utils.exceptions import ConfigAssertion

logger = logging.getLogger(__name__)


class NLPCausalLMConfigUpdater:
    def __init__(self, cfg: ConfigProblemBase):
        self.cfg: ConfigProblemBase = copy_config(cfg)

    def update(self, cfg: ConfigProblemBase):
        """
        Identifies inconsistent configuration items and changes them accordingly, if possible.
        """
        self.update_lora_target_layers(cfg)
        self.update_gpu_ids(cfg)
        self.cfg: ConfigProblemBase = copy_config(cfg)

    def check(self, cfg: ConfigProblemBase) -> dict:
        """
        Checks consistency of the configuration and reports any issues.
        """
        cfg: ConfigProblemBase = copy_config(cfg)

        config_problem_dict = dict()
        gpus = copy(cfg.environment.gpus)
        self.update_gpu_ids(cfg)
        if cfg.environment.gpus != gpus:
            config_problem_dict["gpus"] = \
                "Configuration specifies the following GPU ids: " \
                f"{gpus}, but only found the following GPU ids:" \
                f"{cfg.environment.gpus}. " \
                "This can happen when running an experiment from a configuration " \
                "file that was generated on a machine with more GPUs available. "

        if cfg.training.lora_target_modules is None:
            self.update_lora_target_layers(cfg)
            if cfg.training.lora_target_modules is None:
                config_problem_dict["lora_target_modules"] = "Please specify LORA target modules!"
        return config_problem_dict

    def update_lora_target_layers(self, cfg):
        if self.cfg.llm_backbone != cfg.llm_backbone:
            # update default target modules for lora
            model_type = MODELNAME2MODELTYPE.get(cfg.llm_backbone)
            lora_target_modules = (
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(model_type)
            )
            if lora_target_modules is None:
                # extend LORA automatic target module mapping.
                lora_target_modules = {
                    "RefinedWebModel": [
                        "query_key_value",
                        "dense_h_to_4h",
                        "dense_4h_to_h",
                        "dense",
                    ],
                }.get(model_type, [])
            cfg.training.lora_target_modules = ", ".join(lora_target_modules)

    def update_gpu_ids(self, cfg):
        """
        Check if all gpus in the config are available
        """
        # For better UX, gpu_id start with 1, thus <= in the comparison below
        gpus = tuple(
            gpu_id
            for gpu_id in cfg.environment.gpus
            if gpu_id <= torch.cuda.device_count()
        )
        if gpus != cfg.environment.gpus:
            cfg.environment.gpus = gpus


class ConfigUpdaterFactory:
    """ConfigUpdater factory."""

    _config_updaters = {
        "text_causal_language_modeling_config": NLPCausalLMConfigUpdater,
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._config_updaters.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to ConfigUpdater.

        Args:
            name: problem type name
        Returns:
            A class to build the ConfigUpdater
        """
        return cls._config_updaters.get(name)

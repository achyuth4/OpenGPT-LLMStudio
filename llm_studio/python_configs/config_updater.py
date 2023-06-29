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

__all__ = ["ConfigUpdater"]

from llm_studio.src.utils.exceptions import ConfigAssertion

logger = logging.getLogger(__name__)


class NLPCausalLMConfigUpdater:
    def __init__(self, cfg: ConfigProblemBase):
        self.cfg: ConfigProblemBase = copy_config(cfg)

    def update(self, cfg: ConfigProblemBase):
        self.update_lora_target_layers(cfg)
        self.update_gpu_ids(cfg)
        self.cfg: ConfigProblemBase = copy_config(cfg)

    def check(self, cfg: ConfigProblemBase):
        gpus = copy(cfg.environment.gpus)
        self.update_gpu_ids(cfg)
        if cfg.environment.gpus != gpus:
            raise ConfigAssertion(f"Gpus available {cfg.environment.gpus} is not consistent with"
                                  f"Gpus specified in the configuration {gpus}")

        if self.cfg.training.lora_target_modules is None:
            self.update_lora_target_layers(cfg)
            if self.cfg.training.lora_target_modules is None:
                raise ConfigAssertion(f"Please specify LORA target modules!")
            self.cfg.training.lora_target_modules = None

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
            cfg.training.lora_target_modules = ', '.join(lora_target_modules)

    def update_gpu_ids(self, cfg):
        # For better UX, gpu_id start with 1, thus <= in the comparison below
        gpus = tuple(gpu_id for gpu_id in cfg.environment.gpus
                     if gpu_id <= torch.cuda.device_count())
        if gpus != cfg.environment.gpus:
            logger.warning(f"Configuration specifies the following gpu ids: {cfg.environment.gpus},"
                           f" but only found {torch.cuda.device_count()} number of GPUs. This can happen "
                           f"when running an experiment from a configuration file that was generated on a machine with "
                           f"more GPUs available. Automatically setting GPUs to {gpus}")
            cfg.environment.gpus = gpus


class ConfigUpdater:
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

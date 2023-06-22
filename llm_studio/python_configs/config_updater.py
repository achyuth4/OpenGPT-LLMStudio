from typing import Any, List

from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from app_utils.utils import copy_config
from llm_studio.python_configs.text_causal_language_modeling_config import (
    MODELNAME2MODELTYPE,
    ConfigProblemBase,
)

__all__ = ["ConfigUpdater"]


class NLPCausalLMConfigUpdater:
    def __init__(self, cfg: ConfigProblemBase):
        self.cfg: ConfigProblemBase = copy_config(cfg)

    def __call__(self, cfg: ConfigProblemBase):
        self.update_lora_target_layers(cfg)

        self.cfg: ConfigProblemBase = copy_config(cfg)

    def update_lora_target_layers(self, cfg):
        if self.cfg.llm_backbone != cfg.llm_backbone:
            # update default target modules for lora
            model_type = MODELNAME2MODELTYPE.get(cfg.llm_backbone)
            self.cfg.training.lora_target_modules = (
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(model_type)
            )
            if self.cfg.training.lora_target_modules is None:
                # extend LORA automatic target module mapping.
                self.cfg.training.lora_target_modules = {
                    "RefinedWebModel": [
                        "query_key_value",
                        "dense_h_to_4h",
                        "dense_4h_to_h",
                        "dense",
                    ],
                }.get(model_type)


class ConfigUpdater:
    """ConfigUpdater factory."""

    _config_updaters = {
        "text_causal_language_modeling": NLPCausalLMConfigUpdater,
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

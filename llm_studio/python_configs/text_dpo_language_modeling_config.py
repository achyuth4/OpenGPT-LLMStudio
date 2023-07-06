import os
from dataclasses import dataclass, field
from typing import Any, Tuple

import llm_studio.src.datasets.text_dpo_language_modeling_ds
from llm_studio.python_configs.base import DefaultConfig
from llm_studio.python_configs.text_causal_language_modeling_config import ConfigNLPCausalLMDataset, \
    ConfigNLPCausalLMTokenizer, ConfigNLPCausalLMPrediction, ConfigNLPCausalLMEnvironment, ConfigNLPCausalLMLogging, \
    ConfigNLPCausalLMTraining, ConfigNLPCausalLMArchitecture, ConfigNLPAugmentation
from llm_studio.src import possible_values
from llm_studio.src.losses import text_causal_language_modeling_losses
from llm_studio.src.models import text_causal_language_modeling_model
from llm_studio.src.utils.modeling_utils import generate_experiment_name


@dataclass
class ConfigNLPDPOLMDataset(ConfigNLPCausalLMDataset):
    dataset_class: Any = (
        llm_studio.src.datasets.text_dpo_language_modeling_ds.CustomDataset
    )
    limit_chained_samples: bool = True

    chosen_response_column: str = "chosen_response"
    rejected_response_column: str = "rejected_response"

    def __post_init__(self):
        self.prompt_column = (
            tuple(
                self.prompt_column,
            )
            if isinstance(self.prompt_column, str)
            else tuple(self.prompt_column)
        )
        super().__post_init__()
        self._visibility["limit_chained_samples"] = -1


@dataclass
class ConfigDPOCausalLMTraining(ConfigNLPCausalLMTraining):
    loss_class: Any = text_causal_language_modeling_losses.Losses
    loss_function: str = "TokenAveragedCrossEntropy"
    optimizer: str = "AdamW"


@dataclass
class ConfigDPOCausalLMArchitecture(ConfigNLPCausalLMArchitecture):
    model_class: Any = text_causal_language_modeling_model.Model


@dataclass
class ConfigProblemBase(DefaultConfig):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    _parent_experiment: str = ""
    llm_backbone: str = "EleutherAI/pythia-2.8b-deduped"

    dataset: ConfigNLPDPOLMDataset = field(default_factory=ConfigNLPDPOLMDataset)
    tokenizer: ConfigNLPCausalLMTokenizer = field(
        default_factory=ConfigNLPCausalLMTokenizer
    )
    architecture: ConfigDPOCausalLMArchitecture = field(
        default_factory=ConfigDPOCausalLMArchitecture
    )
    training: ConfigDPOCausalLMTraining = field(
        default_factory=ConfigDPOCausalLMTraining
    )
    augmentation: ConfigNLPAugmentation = field(default_factory=ConfigNLPAugmentation)
    prediction: ConfigNLPCausalLMPrediction = field(
        default_factory=ConfigNLPCausalLMPrediction
    )
    environment: ConfigNLPCausalLMEnvironment = field(
        default_factory=ConfigNLPCausalLMEnvironment
    )
    logging: ConfigNLPCausalLMLogging = field(default_factory=ConfigNLPCausalLMLogging)

    def __post_init__(self):
        super().__post_init__()

        self._visibility["output_directory"] = -1

        self._possible_values["llm_backbone"] = possible_values.String(
            values=(
                "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
                "h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b",
                "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2",
                "tiiuae/falcon-7b",
                "tiiuae/falcon-40b",
                "openlm-research/open_llama_3b",
                "openlm-research/open_llama_7b",
                "openlm-research/open_llama_13b",
                "EleutherAI/gpt-j-6B",
                "EleutherAI/gpt-neox-20b",
                "facebook/opt-125m",
                "facebook/opt-2.7b",
                "EleutherAI/pythia-1b-deduped",
                "EleutherAI/pythia-2.8b-deduped",
                "EleutherAI/pythia-6.9b-deduped",
                "EleutherAI/pythia-12b-deduped",
                "togethercomputer/GPT-NeoXT-Chat-Base-20B",
            ),
            allow_custom=True,
        )

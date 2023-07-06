import codecs
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset as LLMCustomDataset
from llm_studio.src.datasets.text_utils import get_texts, get_tokenizer

logger = logging.getLogger(__name__)


class CustomDataset(LLMCustomDataset):
    """
    Dataset for DPO optimization.
    The data is assumed to be in hierarchical form of the following format:

    Beginning of a chat-answer interaction (parent_id is not set):
        instruction                    What kind of noises did dinosaurs make?
        output               Humans and dinosaurs didn’t live at the same t...
        id                                610e4ad5-09c4-4055-9ff4-948fe6b4f832
        parent_id                                                         None
        chosen_response                                                   None
        rejected_response                                                 None

    Within a chat-answer interaction (parent_id points for the previous prompt-answer sample):
        instruction                                               yes they did
        output               to guess, and that would probably require lots...
        id                                573e8d77-550a-4889-8ff4-1e8d8944897c
        parent_id                         610e4ad5-09c4-4055-9ff4-948fe6b4f832
        chosen_response                                                   None
        rejected_response                                                 None


    Last question. Output should be empty, chosen and rejected responses should be given:
        instruction          Do have a phone number or email address for hi...
        output
        id                                e0edeaf1-166d-4683-8609-dcba6fafc520
        parent_id                         e7e96d54-006d-4b34-a9ed-479c3ec3068c
        chosen_response       He doesn’t have a publicly available phone nu...
        rejected_response     If you want to contact Ryan Reynolds by phone...
    """

    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        """
        Args:
            df: input DataFrame
            cfg: config with all the hyperparameters
            mode: dataset mode. One of {"train", "validation"}
        """

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()

        assert self.mode in [
            "train",
            "validation",
        ], f"There is no {self.mode} for the datasets"

        self.tokenizer = get_tokenizer(cfg)
        self.prompts = [self.parse_prompt(cfg, prompt)
                        for prompt in get_texts(df, self.cfg, separator="")]
        self.answers = (
            self.df[self.cfg.dataset.answer_column].astype(str).values.tolist()
        )
        self.chosen_answers = (
            self.df[self.cfg.dataset.chosen_response_column].astype(str).values.tolist()
        )
        self.rejected_answer = (
            self.df[self.cfg.dataset.rejected_response_column].astype(str).values.tolist()
        )

        self.parent_ids = None
        if self.cfg.dataset.parent_id_column != "None":
            if "id" not in self.df.columns:
                logger.warning(
                    f"When using parent column, the dataframe requires an 'id' column. "
                    f"Disabling functionality for mode {self.mode}."
                )
            else:
                self.parent_ids = self.df[self.cfg.dataset.parent_id_column].values
                self.df_id_to_idx = {v: k for k, v in enumerate(self.df["id"].values)}

                # limit chained samples to the longest chain
                if self.cfg.dataset.limit_chained_samples:
                    self.indices = self.indices[
                        [id not in self.parent_ids for id in self.df["id"].values]
                    ]

        if self.cfg.environment._local_rank == 0:
            logger.info(f"Sample prompt: {self.prompts[0]}")

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        sample = dict()
        encodings = [self._get_prompt_and_answer_encoding(i) for i in self.get_id_chain(idx)]
        assert encodings[-1][1] == []
        input_ids = torch.cat([torch.cat(encoding) for encoding in encodings])

        ####
        sample.update(
            self.pad_tokens(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_length=self.cfg.tokenizer.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        )
        sample.update(
            self.pad_tokens(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_length=self.cfg.tokenizer.max_length_prompt,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix="prompt_",
            )
        )
        return sample

    def get_id_chain(self, idx):
        id_chain = [idx]
        if self.parent_ids is not None:
            parent_idx = idx
            while (
                    parent_idx := self.df_id_to_idx.get(self.parent_ids[parent_idx], None)
            ) is not None:
                id_chain = [parent_idx] + id_chain
        return id_chain

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def get_input_columns(cfg: Any) -> Tuple[str, ...]:
        """Assigns the input columns

        Args:
            cfg: config

        """
        if isinstance(cfg.dataset.prompt_column, tuple):
            return cfg.dataset.prompt_column
        return (cfg.dataset.prompt_column,)

    def postprocess_batch_predictions(self, cfg: Any, output: Dict) -> Dict:
        if cfg.prediction.metric == "Perplexity":
            return output

        predicted_text = [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in output["predicted_answer_ids"]
        ]
        output["predicted_text"] = np.array(predicted_text)

        if not cfg.training.use_rlhf:
            del output["predicted_answer_ids"]
        else:
            output["predicted_answer_ids"].detach()

        return output

    @staticmethod
    def clean_output(
            output: Dict,
            prompts: List[str],
            cfg: Any,
    ):
        output["predicted_text"] = output["predicted_text"].tolist()
        for j in range(len(output["predicted_text"])):
            curr_text = output["predicted_text"][j].strip()
            for stop_token in cfg.tokenizer._stop_words:
                if curr_text.find(stop_token) != -1:
                    curr_text = curr_text[: curr_text.find(stop_token)]
            output["predicted_text"][j] = curr_text.strip()

        return output

    def postprocess_output(self, cfg, df: pd.DataFrame, output: Dict) -> Dict:
        if not cfg.prediction.metric == "Perplexity":
            output = self.clean_output(output, self.prompts, cfg)

        output["target_text"] = self.answers

        metric_func, _, _ = cfg.prediction.metric_class.get(cfg.prediction.metric)

        if "GPT" in cfg.prediction.metric:
            metrics, explanations = metric_func(
                cfg,
                output,
                df,
                raw_results=True,
            )
            output["explanations"] = explanations
        else:
            metrics = metric_func(
                cfg,
                output,
                df,
            )
        output["metrics"] = metrics

        return output

    def format_output(
            self, cfg, df: pd.DataFrame, output: Dict
    ) -> Tuple[Dict, pd.DataFrame]:
        output = {
            key: value
            for key, value in output.items()
            if key not in ["loss", "target", "losses"]
        }

        output.pop("target_text", None)

        if "predicted_text" in output.keys():
            output["predicted_text"] = np.array(output["predicted_text"])

        if isinstance(cfg.dataset.prompt_column, tuple):
            for col in cfg.dataset.prompt_column:
                output[col] = df[col].values
        else:
            output[cfg.dataset.prompt_column] = df[cfg.dataset.prompt_column].values

        if "predicted_text" in output.keys():
            df[f"pred_{cfg.dataset.answer_column}"] = output["predicted_text"]

        return output, df

if __name__ == '__main__':
    df = pd.read_parquet("/home/max/PycharmProjects/h2o-llmstudio/data/hh.pq")
    df.head()

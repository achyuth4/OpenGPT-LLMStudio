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

        self.indices = np.arange(len(self.df))

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
        self.chosen_response = (
            self.df[self.cfg.dataset.chosen_response_column].astype(str).values.tolist()
        )
        self.rejected_response = (
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
        prompt_encoding, answer_encoding = self._get_prompt_and_answer_encoding(idx)
        encodings = [[prompt_encoding, answer_encoding]]
        parent_encodings, _ = self.get_parent_encodings(idx)
        encodings = parent_encodings + encodings

        ####
        # Add preferred and rejected answer

        ####

        input_ids = torch.cat([torch.cat(encoding) for encoding in encodings])
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

    @staticmethod
    def parse_prompt(cfg: Any, prompt: str):
        prompt = (
            f"{codecs.decode(cfg.dataset.text_prompt_start, 'unicode_escape')}{prompt}"
        )
        if cfg.dataset.add_eos_token_to_prompt:
            prompt += cfg._tokenizer_eos_token
        prompt = (
            f"{prompt}"
            f"{codecs.decode(cfg.dataset.text_answer_separator, 'unicode_escape')}"
        )
        return prompt

    def __len__(self) -> int:
        return len(self.df)

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

    def _get_prompt_and_answer_encoding(self, idx) -> List:
        prompt = self.prompts[idx]
        answer = self.answers[idx]

        prompt_encoding = self.encode(
            self.tokenizer, prompt, self.cfg.tokenizer.max_length_prompt, "left"
        )["input_ids"]
        if self.cfg.dataset.add_eos_token_to_answer:
            max_length_answer = self.cfg.tokenizer.max_length_answer - 1
        else:
            max_length_answer = self.cfg.tokenizer.max_length_answer
        answer_encoding = self.encode(
            self.tokenizer, answer, max_length_answer, "right"
        )["input_ids"]
        if self.cfg.dataset.add_eos_token_to_answer:
            answer_encoding = torch.cat(
                [
                    answer_encoding,
                    torch.Tensor([self.tokenizer.eos_token_id]),
                ],
                dim=0,
            )

        return [prompt_encoding, answer_encoding]


if __name__ == '__main__':
    df = pd.read_parquet("/home/max/PycharmProjects/h2o-llmstudio/data/hh.pq")
    df.head()

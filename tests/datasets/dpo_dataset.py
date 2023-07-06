import pandas as pd

from llm_studio.python_configs.text_dpo_language_modeling_config import ConfigProblemBase, ConfigNLPDPOLMDataset
from llm_studio.src.datasets.text_dpo_language_modeling_ds import CustomDataset

if __name__ == "__main__":
    df = pd.read_parquet("/home/max/PycharmProjects/h2o-llmstudio/data/hh.pq")
    df.head()

    cfg = ConfigProblemBase(dataset=ConfigNLPDPOLMDataset(prompt_column=("instruction",),
                                                          answer_column="output",
                                                          parent_id_column="parent_id"
                                                          ))

    dataset = CustomDataset(df, cfg, mode='train')
    print(len(dataset))
    print(dataset[0])

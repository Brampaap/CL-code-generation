import os
from pathlib import Path

import lightning.pytorch as pl
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from yaml.loader import SafeLoader

from CL.common.dataset import utils as dataset_utils
from CL.common.dataset.pauq.pauq_dataset import Text2SQLDataset
from CL.common.tokenizer import Tokenizer
from CL.common.utils import collate_fn
from CL.curriculum import difficulty_sorter, metrics

CONFIG = yaml.load(
    (Path(os.environ["PROJECT_PATH"]) / "config.yaml").open(),
    Loader=SafeLoader,
)
LCQUAD_CONFIG = CONFIG["dataset"]["lcquad"]
MODEL_CONFIG = CONFIG["models"]["seq2seqGRU"]
CL_CONFIG = CONFIG["curriculum"]


def extract_tokenized(
    dataset: list[dict],
) -> tuple[list[list[str]], list[list[str]]]:
    tokenized_inputs = list()
    tokenized_target = list()

    for item in dataset:
        tokenized_inputs.append(item["question"])
        tokenized_target.append(item["masked_query"])

    return tokenized_inputs, tokenized_target


class Text2SPARQLDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super(Text2SPARQLDataModule, self).__init__()
        self.cl_train_dataset: Dataset | None = None
        self.batch_size = batch_size
        self.src_tokenizer: Tokenizer | None = None
        self.trg_tokenizer: Tokenizer | None = None

    @staticmethod
    def get_tokenizers(train_data: list[dict]):
        (
            pre_tokenized_input,
            pre_tokenized_targets,
        ) = extract_tokenized(train_data)
        text_tokenizer = Tokenizer(dataset=pre_tokenized_input)
        sql_tokenizer = Tokenizer(dataset=pre_tokenized_targets)
        return text_tokenizer, sql_tokenizer

    def __get_train_dev_test__(
        self,
        train,
        test,
    ) -> list[Text2SQLDataset]:
        assert (
            self.src_tokenizer is not None or self.trg_tokenizer is not None
        ), "Tokenizers was not defined."

        train_proportion = round(
            (1.0 - LCQUAD_CONFIG["train"]["dev_size"]) * len(train)
        )
        proportions = [train_proportion, len(train) - train_proportion]

        train_splitted, dev = random_split(dataset=train, lengths=proportions)

        processed = []
        for dataset in [train_splitted, dev, test]:
            (
                pre_tokenized_input,
                pre_tokenized_targets,
            ) = extract_tokenized(dataset)

            tokenized_input: list[list[int]] = self.src_tokenizer.encode_all(
                pre_tokenized_input
            )
            tokenized_targets: list[list[int]] = self.trg_tokenizer.encode_all(
                pre_tokenized_targets
            )
            processed.append(
                Text2SQLDataset(
                    src_data=tokenized_input, trg_data=tokenized_targets
                )
            )

        return processed

    def make_curriculum_step(self, fraction: float):
        n_elements = int(len(self.train_dataset) * fraction)
        self.cl_train_dataset = Text2SQLDataset(
            self.train_dataset.src_data[:n_elements],
            self.train_dataset.trg_data[:n_elements],
        )

    def setup(self, stage=None):
        train_full, test = dataset_utils.load(LCQUAD_CONFIG)
        self.src_tokenizer, self.trg_tokenizer = self.get_tokenizers(
            train_full
        )
        (
            self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
        ) = self.__get_train_dev_test__(train_full, test)

        metric = metrics.SentLenMetric()
        self.train_dataset = difficulty_sorter.sort_dataset(
            metric, self.train_dataset
        )

        self.make_curriculum_step(CL_CONFIG["min_fraction"])

    def train_dataloader(self):
        return DataLoader(
            self.cl_train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=1,
        )

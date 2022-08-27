"""
Helpers for CSQA using huggingface/transformers and torch_geometric

csqa_text = (
    csqa_df
    .explode("choices")
    .rename({"choices": "answer"}, axis=1)
    .to_records()
)
csqa_graph = get_graphs(csqa_text)
dataset = BatchDataset(ZipDataset([csqa_text, csqa_graph]), batch_size=num_choices)
dataloader = DataLoader(
    dataset,
    collate_fn=BatchCollator(ZipCollator([TextCollator(tokenizer), GraphCollator]))
)
"""

from typing import Callable, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch_geometric.data import Batch


BERT_FEATURE_NAMES = ["input_ids", "token_type_ids", "attention_mask"]
GPT_FEATURE_NAMES = ["input_ids", "attention_mask"]


@dataclass
class BatchDataset(Dataset):
    ds: Dataset
    batch_size: int

    def __len__(self):
        return len(self.ds) // self.batch_size

    def __getitem__(self, i):
        return [self.ds[i * self.batch_size + k]
                for k in range(self.batch_size)]


@dataclass
class ZipDataset(Dataset):
    ds: List[Dataset]

    def __len__(self):
        return len(self.ds[0])

    def __getitem__(self, i):
        return tuple(ds[i] for ds in self.ds)


@dataclass
class BatchCollator:
    wrapped_collator: Callable

    def __call__(self, batch):
        flattened = sum(batch, [])
        return self.wrapped_collator(flattened)


@dataclass
class ZipCollator:
    collators: List[Callable]

    def __call__(self, batch):
        *batches, = zip(*batch)
        return tuple(collate(batch) for collate, batch in zip(self.collators, batches))


@dataclass
class TextCollator:
    tokenizer: PreTrainedTokenizerBase
    labels = True
    features = ["question", "answer"]  # at most 2

    def __call__(self, records):
        input = [[x[feat] for x in records] for feat in self.features]
        tokenized = self.tokenizer(*input, padding=True, return_tensors="pt")

        if self.labels:
            labels = [x["label"] for x in records]
            tokenized["labels"] = torch.tensor(labels, dtype=torch.int64)

        return tokenized


GraphCollator = Batch.from_data_list

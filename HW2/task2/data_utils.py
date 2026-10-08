from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from collections import defaultdict
import random


MAX_LENGTH = 512


class BaseDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for line in data_file:
                if line.strip():
                    self.samples.append(line)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        el = self.samples[idx]
        input_ids = self.tokenizer(el)['input_ids']
        input_ids = input_ids[:self.max_length]
        length = len(input_ids)
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        return torch.tensor(input_ids, dtype=torch.int64), length


class StandardDataset(Dataset):
    """
    See task desciption
    """
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = tokenizer
        samples = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for line in data_file:
                if line.strip():
                    samples.append(line)
        self.samples = [input_ids[:max_length] for input_ids in tokenizer(samples)['input_ids']]
        self.lengths = [len(input_ids) for input_ids in self.samples]
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return torch.tensor(self.samples[idx], dtype=torch.int64), self.lengths[idx]


class SequencedDataset(IterableDataset):
    """
    See task desciption
    """
    def __init__(
        self, 
        data_path: str,
        tokenizer: AutoTokenizer,
        batch_size: int, 
        max_length: int = MAX_LENGTH
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        samples = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for line in data_file:
                if line.strip():
                    samples.append(line)
        self.samples = [input_ids[:max_length] for input_ids in tokenizer(samples)['input_ids']]
    
    def __iter__(self):
        order = torch.randperm(len(self.samples)).tolist()
        pack, pack_length = [], 0
        for idx in order:
            sample = self.samples[idx]
            if pack and (pack_length + len(sample) > self.max_length or len(pack) == self.batch_size):
                yield pack
                pack, pack_length = [], 0
            pack.append(sample)
            pack_length += len(sample)
        if pack:
            yield pack

    def collate_data(self, batch):
        lengths = torch.tensor([len(sample) for sample in batch])
        tokens = torch.tensor([token for sample in batch for token in sample], dtype=torch.int64)
        total_length = tokens.shape[0]
        sample_ids = torch.repeat_interleave(torch.arange(len(batch)), lengths)
        causal = torch.tril(torch.ones((total_length, total_length), dtype=torch.bool))
        same_sample = sample_ids[:, None] == sample_ids[None, :]
        attention_mask = causal & same_sample
        loss_mask = (sample_ids[:-1] == sample_ids[1:]).unsqueeze(0)
        return {
            'tokens': tokens.unsqueeze(0),
            'lengthes': torch.tensor([total_length]),
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
        }


def base_collate_fn(
    batch: list[tuple[str, torch.Tensor]],
    pad_token_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_processed = torch.stack([el[0] for el in batch])
    lengthes = torch.tensor([el[1] for el in batch])
    return {
        'tokens': batch_processed, 
        'lengthes': lengthes, 
        'attention_mask': None,
        'loss_mask': None,
    }


def collate_fn(
    batch: list[tuple[str, torch.Tensor]],
    pad_token_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    See task desciption
    """
    mx_len = 0
    for el in batch:
        mx_len = max(mx_len, el[1])
    batch_processed = []
    for el in batch:
        processed_el = torch.cat((el[0], torch.full((mx_len - el[1],), pad_token_id)), dim=0)
        batch_processed.append(processed_el)
    batch_processed = torch.stack(batch_processed)
    lengthes = torch.tensor([el[1] for el in batch]) 
    return {
        'tokens': batch_processed, 
        'lengthes': lengthes, 
        'attention_mask': None,
        'loss_mask': None,
    }


class BalancedBatchSampler(Sampler):
    """
    See task desciption
    """
    def __init__(self, dataset, k: int, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.buckets = defaultdict(list)
        for idx, length in enumerate(dataset.lengths):
            self.buckets[length // (k + 1)].append(idx)
        self.keys = list(self.buckets)
        self.sizes = torch.tensor([len(self.buckets[key]) for key in self.keys])
        self.num_batches = int(torch.ceil(self.sizes / batch_size).sum())

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        remaining = self.sizes.clone()
        for _ in range(self.num_batches):
            bucket_id = torch.multinomial(remaining.float(), 1).item()
            indices = self.buckets[self.keys[bucket_id]]
            left = remaining[bucket_id].item()
            batch = []
            for _ in range(min(self.batch_size, left)):
                position = torch.randint(left, (1,)).item()
                left -= 1
                indices[position], indices[left] = indices[left], indices[position]
                batch.append(indices[left])
            remaining[bucket_id] = left
            yield batch

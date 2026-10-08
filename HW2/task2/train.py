import math
import platform
import random
import time
import gc
from collections import defaultdict
from functools import partial
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, Sampler
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from data_utils import BaseDataset, StandardDataset, SequencedDataset, base_collate_fn, collate_fn, BalancedBatchSampler
from positional_encoding import PositionalEncoding
from amp import Autocast, StaticGradScaler, DynamicGradScaler
from utils import LMCrossEntropyLoss, LMAccuracy


MAX_LENGTH = 512
TOKENIZER_PATH = "bert-base-uncased"  # Или путь к заранее добавленному tokenizer.
INPUT_DIR = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()

data_candidates = sorted(INPUT_DIR.rglob("validation-00000-of-00001.txt")) if INPUT_DIR.is_dir() else []
DATA_PATH = "/kaggle/input/datasets/axilles60/wikitext-103-raw-v1/wikitext-103-raw-v1/validation-00000-of-00001.txt"
# Если нужно, замените DATA_PATH на точный путь к файлу из Add Input.

CONFIG = SimpleNamespace(
    kind="fp32",          # fp32 | fp16 | static | dynamic
    batch_size=32,
    num_epochs=1,
    scale=65536.0,
    factor=2.0,
    patience=80,
    min_scale=1.0,
    max_scale=2.0**24,
    dataloader="base",     # base | standard | balanced | sequenced
    path=DATA_PATH,
    k=20,
)
RUN_TRAINING = True

print("Параметры:", vars(CONFIG))
print("WORK_DIR:", WORK_DIR)
if len(data_candidates) > 1:
    print("Найдено несколько файлов; укажите DATA_PATH явно:", *data_candidates, sep="\n")
if not Path(CONFIG.path).is_file():
    print("Файл датасета пока не найден. Добавьте Input и задайте DATA_PATH перед обучением.")


def set_global_seed(seed: int) -> None:
    """
    Set global seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class GPT2LikeModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=1020, num_heads=4, max_length=MAX_LENGTH):
        super().__init__()
        self.num_heads = num_heads
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max_length)
        self.hidden_projector = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_projector2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.output_linear = torch.nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, attention_mask):
        x = x.transpose(0, 1) # as we don't use batch first
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.hidden_projector(x)
        y = self.hidden_projector2(x)
        if attention_mask is None:
            attention_mask = torch.tril(
                torch.ones((x.size(0), x.size(0)), dtype=torch.bool, device=x.device)
            )
        else:
            attention_mask = attention_mask.to(x.device)
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros_like(attention_mask, dtype=torch.float32).masked_fill_(
                attention_mask.logical_not(), float("-inf")
            )
        attention_mask = attention_mask.to(x.dtype)

        out = self.decoder(tgt=x, memory=x, tgt_mask=attention_mask, memory_mask=attention_mask)
        out = self.output_linear(out)
        return out.transpose(0, 1)


def get_gpt2_model(vocab_size) -> torch.nn.Module:
    return GPT2LikeModel(vocab_size)


def get_dataloader(dataloader_type, batch_size, path, k):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    dataloader = None
    if dataloader_type == 'base':
        ds = BaseDataset(path, tokenizer)
        collate = partial(base_collate_fn, pad_token_id=tokenizer.pad_token_id)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    if dataloader_type == 'standard':
        ds = StandardDataset(path, tokenizer)
        collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    if dataloader_type == 'balanced':
        ds = StandardDataset(path, tokenizer)
        collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
        sampler = BalancedBatchSampler(ds, k=k, batch_size=batch_size)
        dataloader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)
    if dataloader_type == 'sequenced':
        ds = SequencedDataset(path, tokenizer, batch_size=batch_size)
        dataloader = DataLoader(ds, batch_size=None, collate_fn=ds.collate_data)
    return dataloader


def train_epoch(
    train_loader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    metric: Callable, 
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kind: str,
    scaler: None | StaticGradScaler | DynamicGradScaler,
    profiler=None,
    max_steps=None,
) -> dict[str, list[float]]:
    model.train()
    autocast = Autocast(enabled=kind in ('static', 'dynamic'))
    stats = {'loss': [], 'accuracy': [], 'grad_time': [], 'scale': []}
    pbar = tqdm(enumerate(train_loader))
    for i, data in pbar:
        tokens, tokens_lens, attention_mask = data['tokens'].to(device), data['lengthes'], data['attention_mask']
        loss_mask = data.get('loss_mask')
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)

        optimizer.zero_grad(set_to_none=False)
        # Obtain outputs and loss depending on kind. Plain fp16 should run without
        # Autocast; static/dynamic modes should use the custom Autocast.
        # Pass loss_mask to the criterion for sequenced batches.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with autocast:
            outputs = model(tokens, attention_mask)
            loss = criterion(outputs, tokens, tokens_lens, loss_mask=loss_mask)

        if kind == 'fp16' or kind == 'fp32':
            # compute grads without scaling
            loss.backward()
        else:
            # compute grads with scaling
            scaler.scale(loss).backward()
        end.record()
        torch.cuda.synchronize() 
        stats['grad_time'].append(start.elapsed_time(end) / 1000)
        stats['scale'].append(1.0 if scaler is None else scaler._scale)

        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        accuracy = metric(outputs, tokens, tokens_lens, loss_mask=loss_mask)

        loss_value, accuracy_value = loss.item(), accuracy.item()
        stats['loss'].append(loss_value)
        stats['accuracy'].append(accuracy_value)
        pbar.set_description(f"Loss: {round(loss_value, 4)} " f"Accuracy: {round(accuracy_value * 100, 4)}")
        if profiler is not None:
            profiler.step()
        if max_steps is not None and i + 1 >= max_steps:
            break

    return stats


def train(args=None, profiler=None, max_steps=None):
    set_global_seed(42)
    args = CONFIG if args is None else args
    if not torch.cuda.is_available():
        raise RuntimeError("Для обучения включите GPU в настройках Kaggle.")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    prep_start = time.perf_counter()
    dataloader = get_dataloader(args.dataloader, args.batch_size, args.path, args.k)
    prep_time = time.perf_counter() - prep_start
    model = get_gpt2_model(dataloader.dataset.tokenizer.vocab_size).to(device)
    if args.kind == 'fp16':
        model = model.half()
    criterion = LMCrossEntropyLoss()
    metric = LMAccuracy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = None
    if args.kind == 'static':
        scaler = StaticGradScaler(args.scale)
    elif args.kind == 'dynamic':
        scaler = DynamicGradScaler(
            scale=args.scale,
            factor=args.factor,
            patience=args.patience,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
    num_epochs = args.num_epochs
    stats = {'loss': [], 'accuracy': [], 'grad_time': [], 'scale': []}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for epoch in range(0, num_epochs):
        epoch_stats = train_epoch(
            train_loader=dataloader, 
            model=model, 
            criterion=criterion, 
            metric=metric,
            optimizer=optimizer, 
            device=device, 
            kind=args.kind, 
            scaler=scaler,
            profiler=profiler,
            max_steps=max_steps,
        )
        first_step = len(stats['loss'])
        for key, values in epoch_stats.items():
            stats[key].extend(values)
    end.record()
    torch.cuda.synchronize()
    stats['prep_time'] = prep_time
    stats['train_time'] = start.elapsed_time(end) / 1000
    stats['peak_memory_mb'] = torch.cuda.max_memory_allocated(device) / 2**20
    return stats


if __name__ == '__main__':
    train()

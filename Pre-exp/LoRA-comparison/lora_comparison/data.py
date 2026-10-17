"""Pinned WikiText-2 and PIQA inputs for teacher-free adaptation/evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .protocol import PIQA_DATASET, PIQA_REVISION, WIKITEXT_REVISION


@dataclass(frozen=True)
class ChoiceExample:
    choices: tuple[list[int], list[int]]
    continuation_starts: tuple[int, int]
    label: int


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def load_wikitext_blocks(
    tokenizer,
    *,
    split: str,
    block_length: int,
    local_files_only: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Concatenate WikiText-2 and return non-overlapping token blocks."""
    if local_files_only:
        # datasets honors offline environment variables; download_mode avoids
        # refreshing metadata while still using the normal cache resolver.
        download_mode = "reuse_dataset_if_exists"
    else:
        download_mode = "reuse_dataset_if_exists"
    dataset = _load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        revision=WIKITEXT_REVISION,
        download_mode=download_mode,
    )
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("tokenizer needs an EOS token")
    token_ids: list[int] = []
    nonempty = 0
    for text in dataset["text"]:
        if not text or not text.strip():
            continue
        nonempty += 1
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        token_ids.append(eos)
    usable = len(token_ids) // block_length * block_length
    if usable == 0:
        raise RuntimeError(f"WikiText {split} did not produce one {block_length}-token block")
    blocks = torch.tensor(token_ids[:usable], dtype=torch.long).view(-1, block_length)
    return blocks, {
        "dataset": "wikitext/wikitext-2-raw-v1",
        "revision": WIKITEXT_REVISION,
        "split": split,
        "rows": len(dataset),
        "nonempty_rows": nonempty,
        "tokens": usable,
        "blocks": len(blocks),
        "block_length": block_length,
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }


def _encode_piqa(tokenizer, row: dict[str, Any], max_length: int) -> ChoiceExample:
    prompt = f"Question: {row['goal']}\nAnswer:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    encoded: list[list[int]] = []
    starts: list[int] = []
    for key in ("sol1", "sol2"):
        continuation = tokenizer.encode(" " + row[key], add_special_tokens=False)
        # Preserve the complete answer and truncate old prompt context first.
        keep_prompt = max(1, max_length - len(continuation))
        local_prompt = prompt_ids[-keep_prompt:]
        sequence = (local_prompt + continuation)[:max_length]
        start = len(local_prompt)
        if start >= len(sequence):
            raise RuntimeError("PIQA continuation vanished after truncation")
        encoded.append(sequence)
        starts.append(start)
    label = int(row["label"])
    if label not in (0, 1):
        raise RuntimeError(f"invalid PIQA label {label}")
    return ChoiceExample((encoded[0], encoded[1]), (starts[0], starts[1]), label)


def load_piqa_examples(
    tokenizer,
    *,
    split: str,
    max_length: int,
    limit: int | None = None,
) -> tuple[list[ChoiceExample], dict[str, Any]]:
    dataset = _load_dataset(
        PIQA_DATASET,
        split=split,
        revision=PIQA_REVISION,
        download_mode="reuse_dataset_if_exists",
    )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    examples = [_encode_piqa(tokenizer, row, max_length) for row in dataset]
    return examples, {
        "dataset": PIQA_DATASET,
        "revision": PIQA_REVISION,
        "split": split,
        "examples": len(examples),
        "max_length": max_length,
        "prompt": "Question: {{goal}}\\nAnswer:",
        "choices": ["sol1", "sol2"],
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }


def collate_choice_examples(
    examples: list[ChoiceExample],
    *,
    pad_token_id: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    flattened: list[list[int]] = []
    starts: list[int] = []
    labels: list[int] = []
    for example in examples:
        flattened.extend(example.choices)
        starts.extend(example.continuation_starts)
        labels.append(example.label)
    length = max(map(len, flattened))
    input_ids = torch.full((len(flattened), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    continuation_mask = torch.zeros((len(flattened), length - 1), dtype=torch.bool)
    for row, (tokens, start) in enumerate(zip(flattened, starts)):
        input_ids[row, : len(tokens)] = torch.tensor(tokens)
        attention_mask[row, : len(tokens)] = 1
        # logits at position p predict token p+1.
        continuation_mask[row, max(start - 1, 0) : len(tokens) - 1] = True
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "continuation_mask": continuation_mask.to(device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def shuffled_minibatches(
    length: int,
    batch_size: int,
    *,
    seed: int,
) -> Iterator[list[int]]:
    """Yield deterministic shuffled batches forever, advancing the epoch seed."""
    if length <= 0:
        raise ValueError("dataset is empty")
    epoch = 0
    while True:
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(length, generator=generator).tolist()
        for start in range(0, length, batch_size):
            batch = order[start : start + batch_size]
            if len(batch) == batch_size:
                yield batch
        epoch += 1

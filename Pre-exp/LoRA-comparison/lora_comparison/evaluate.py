"""Matched WikiText-2 perplexity and PIQA acc_norm evaluation."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .data import collate_choice_examples, load_piqa_examples, load_wikitext_blocks
from .modeling import RoutedGroupMLP, count_parameters
from .objectives import normalized_choice_scores
from .protocol import RunConfig, VariantSpec, save_json


@torch.no_grad()
def evaluate_wikitext(model, tokenizer, config: RunConfig) -> dict[str, Any]:
    blocks, identity = load_wikitext_blocks(
        tokenizer,
        split="test",
        block_length=config.max_length,
        local_files_only=config.local_files_only,
    )
    if config.wikitext_eval_blocks is not None:
        blocks = blocks[: config.wikitext_eval_blocks]
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(blocks), config.eval_batch_size):
        input_ids = blocks[start : start + config.eval_batch_size].to(config.device)
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        ).logits
        targets = input_ids[:, 1:]
        total_nll += float(
            F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
        )
        total_tokens += targets.numel()
    mean_nll = total_nll / total_tokens
    return {
        "metric": "perplexity",
        "score": math.exp(min(mean_nll, 80.0)),
        "mean_nll": mean_nll,
        "evaluated_tokens": total_tokens,
        "evaluated_blocks": len(blocks),
        "data": identity,
    }


@torch.no_grad()
def evaluate_piqa(model, tokenizer, config: RunConfig) -> dict[str, Any]:
    examples, identity = load_piqa_examples(
        tokenizer,
        split="validation",
        max_length=config.max_length,
        limit=config.piqa_eval_limit,
    )
    correct = 0
    total = 0
    pad = tokenizer.pad_token_id
    if pad is None:
        raise RuntimeError("tokenizer needs a pad token")
    for start in range(0, len(examples), config.eval_batch_size):
        batch_examples = examples[start : start + config.eval_batch_size]
        batch = collate_choice_examples(
            batch_examples, pad_token_id=pad, device=config.device
        )
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
        scores = normalized_choice_scores(
            logits, batch["input_ids"], batch["continuation_mask"]
        )
        correct += int((scores.argmax(-1) == batch["labels"]).sum())
        total += len(batch_examples)
    return {
        "metric": "acc_norm",
        "score": correct / total,
        "correct": correct,
        "examples": total,
        "data": identity,
    }


def router_diagnostics(model) -> dict[str, Any] | None:
    entropies = [
        module.last_router_entropy
        for module in model.modules()
        if isinstance(module, RoutedGroupMLP) and module.last_router_entropy is not None
    ]
    if not entropies:
        return None
    return {
        "last_batch_mean_layer_entropy": sum(entropies) / len(entropies),
        "layers": len(entropies),
    }


def evaluate_model(
    model,
    tokenizer,
    *,
    config: RunConfig,
    variant: VariantSpec,
    size: str,
    train_dataset: str,
    tasks: list[str],
    output_path: Path,
    compression_metadata: dict[str, Any],
) -> dict[str, Any]:
    unknown = sorted(set(tasks) - {"wikitext", "piqa"})
    if unknown:
        raise ValueError(f"unknown evaluation tasks {unknown}")
    model.eval()
    model.config.use_cache = False
    started = time.time()
    results: dict[str, Any] = {}
    if "wikitext" in tasks:
        results["wikitext"] = evaluate_wikitext(model, tokenizer, config)
    if "piqa" in tasks:
        results["piqa"] = evaluate_piqa(model, tokenizer, config)
    payload = {
        "schema_version": 1,
        "size": size,
        "variant": variant.key,
        "variant_label": variant.label,
        "train_dataset": train_dataset,
        "teacher_free": True,
        "knowledge_distillation": False,
        "parameters": count_parameters(model),
        "adapter_rank": config.rank if variant.trainable else 0,
        "num_groups": compression_metadata["compression"]["num_groups"],
        "selected_experts": compression_metadata["compression"]["selected_experts"],
        "metrics": results,
        "router": router_diagnostics(model),
        "evaluation_seed": config.eval_seed,
        "elapsed_seconds": time.time() - started,
    }
    save_json(output_path, payload)
    return payload

"""Teacher-free LoRA adaptation on WikiText-2 or PIQA."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import (
    collate_choice_examples,
    load_piqa_examples,
    load_wikitext_blocks,
    shuffled_minibatches,
)
from .modeling import save_trainable_state, trainable_state, variant_metadata
from .objectives import piqa_supervised_loss
from .protocol import RunConfig, VariantSpec, canonical_json_sha256, save_json


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_factor(config: RunConfig, step: int) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if step < config.warmup_steps:
        return (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(
        config.train_steps - config.warmup_steps, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    minimum = config.min_lr / config.learning_rate
    return minimum + (1.0 - minimum) * cosine


def _checkpoint(
    model,
    optimizer,
    scheduler,
    run_dir: Path,
    step: int,
    metadata: dict[str, Any],
) -> None:
    destination = run_dir / "checkpoints" / f"step_{step:06d}"
    save_trainable_state(model, destination, metadata={**metadata, "step": step})
    payload = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = destination / ".trainer_state.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(destination / "trainer_state.pt")


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    for path in (run_dir / "checkpoints").glob("step_*"):
        try:
            step = int(path.name.split("_")[-1])
        except ValueError:
            continue
        if (path / "adapter.safetensors").is_file() and (path / "trainer_state.pt").is_file():
            checkpoints.append((step, path))
    return max(checkpoints, default=(0, None))[1]


def _restore_checkpoint(model, optimizer, scheduler, path: Path) -> int:
    from .modeling import load_trainable_state

    load_trainable_state(model, path)
    payload = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["step"])


def train_adapter(
    model,
    tokenizer,
    *,
    config: RunConfig,
    variant: VariantSpec,
    size: str,
    train_dataset: str,
    run_dir: Path,
    compression_metadata: dict[str, Any],
    resume: bool = True,
) -> dict[str, Any]:
    if not variant.trainable:
        raise ValueError(f"{variant.key} is an evaluation-only baseline")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"{variant.key} should be trainable but has no trainable parameters")
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "size": size,
        "variant": variant_metadata(variant, config.rank, config.alpha),
        "train_dataset": train_dataset,
        "training": {
            "objective": (
                "causal language-model cross entropy"
                if train_dataset == "wikitext"
                else "PIQA normalized two-choice cross entropy"
            ),
            "teacher_free": True,
            "knowledge_distillation": False,
            "steps": config.train_steps,
            "warmup_steps": config.warmup_steps,
            "peak_learning_rate": config.learning_rate,
            "min_learning_rate": config.min_lr,
            "batch_size": config.batch_size,
            "micro_batch_size": config.micro_batch_size,
            "max_length": config.max_length,
            "seed": config.seed,
        },
        "compression_plan_sha256": compression_metadata["plan_sha256"],
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        from .protocol import load_json

        old = load_json(contract_path)
        if old.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(f"run contract changed for existing directory {run_dir}")
    else:
        save_json(contract_path, contract)

    pad = tokenizer.pad_token_id
    if pad is None:
        raise RuntimeError("tokenizer needs a pad token")
    if train_dataset == "wikitext":
        training_data, data_identity = load_wikitext_blocks(
            tokenizer,
            split="train",
            block_length=config.max_length,
            local_files_only=config.local_files_only,
        )
    elif train_dataset == "piqa":
        training_data, data_identity = load_piqa_examples(
            tokenizer,
            split="train",
            max_length=config.max_length,
        )
    else:
        raise ValueError(f"unknown train dataset {train_dataset!r}")
    save_json(run_dir / "data_identity.json", data_identity)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_factor(config, step)
    )
    start_step = 0
    if resume:
        latest = _latest_checkpoint(run_dir)
        if latest is not None:
            start_step = _restore_checkpoint(model, optimizer, scheduler, latest)
            print(f"resumed {run_dir.name} at step {start_step}", flush=True)

    accumulation = config.batch_size // config.micro_batch_size
    # The sequence of sampled examples is a pure function of the optimizer
    # step, so resuming does not silently alter the data order.
    batch_iterator = shuffled_minibatches(
        len(training_data), config.micro_batch_size, seed=config.seed
    )
    for _ in range(start_step * accumulation):
        next(batch_iterator)

    log_path = run_dir / "training.jsonl"
    log_mode = "a" if start_step else "w"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    history_last: dict[str, Any] | None = None
    with log_path.open(log_mode, encoding="utf-8") as log:
        for step_index in range(start_step, config.train_steps):
            losses = 0.0
            accuracies = 0.0
            for _micro in range(accumulation):
                indices = next(batch_iterator)
                if train_dataset == "wikitext":
                    input_ids = training_data[indices].to(config.device)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        labels=input_ids,
                        use_cache=False,
                    )
                    loss = output.loss
                    with torch.no_grad():
                        prediction = output.logits[:, :-1].argmax(-1)
                        accuracy = (prediction == input_ids[:, 1:]).float().mean()
                else:
                    examples = [training_data[index] for index in indices]
                    batch = collate_choice_examples(
                        examples, pad_token_id=pad, device=config.device
                    )
                    loss, accuracy = piqa_supervised_loss(model, batch)
                (loss / accumulation).backward()
                losses += float(loss.detach()) / accumulation
                accuracies += float(accuracy.detach()) / accumulation
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step = step_index + 1
            record = {
                "step": step,
                "loss": losses,
                "train_accuracy": accuracies,
                "grad_norm": grad_norm,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            history_last = record
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            if step == 1 or step % config.log_every == 0 or step == config.train_steps:
                print(
                    f"[{size}/{variant.key}/{train_dataset}] {step}/{config.train_steps} "
                    f"loss={losses:.5f} acc={accuracies:.4f} grad={grad_norm:.3f}",
                    flush=True,
                )
            if step % config.save_every == 0 and step < config.train_steps:
                _checkpoint(model, optimizer, scheduler, run_dir, step, contract)

    final_dir = run_dir / "final"
    save_trainable_state(
        model,
        final_dir,
        metadata={**contract, "completed_steps": config.train_steps},
    )
    summary = {
        "completed": True,
        "steps": config.train_steps,
        "last": history_last,
        "elapsed_seconds": time.time() - started,
        "trainable_tensor_count": len(trainable_state(model)),
        "data": data_identity,
    }
    save_json(run_dir / "training_summary.json", summary)
    model.eval()
    return summary

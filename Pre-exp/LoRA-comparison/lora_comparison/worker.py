"""One-process worker so each large model is released between conditions."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .evaluate import evaluate_model
from .modeling import (
    count_parameters,
    install_variant,
    load_trainable_state,
)
from .protocol import RunConfig, SIZE_SPECS, VARIANTS, load_json
from .spread import backbone_complete, backbone_dir
from .train import seed_everything, train_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=("train", "evaluate", "train-evaluate"), required=True)
    parser.add_argument("--size", choices=tuple(SIZE_SPECS), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--train-dataset", choices=("none", "wikitext", "piqa"), required=True)
    parser.add_argument("--eval-tasks", nargs="+", choices=("wikitext", "piqa"), default=["wikitext", "piqa"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(**load_json(args.config)).validate()
    spec = SIZE_SPECS[args.size]
    variant = VARIANTS[args.variant]
    if variant.trainable and args.train_dataset == "none":
        raise ValueError(f"{variant.key} needs --train-dataset wikitext or piqa")
    if not variant.trainable and args.train_dataset != "none":
        raise ValueError(f"{variant.key} is an evaluation-only baseline; use dataset 'none'")
    if not backbone_complete(config, args.size):
        raise FileNotFoundError(f"incomplete backbone: {backbone_dir(config, args.size)}")

    seed_everything(config.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = backbone_dir(config, args.size)
    dtype = getattr(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=dtype,
        device_map={"": config.device},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    router_init = torch.load(source / "router_init.pt", map_location="cpu", weights_only=True)
    install_variant(
        model,
        variant,
        num_groups=spec.num_groups,
        rank=config.rank,
        alpha=config.alpha,
        router_init=router_init,
    )
    counts = count_parameters(model)
    print(
        f"{args.size}/{variant.key}: {counts['trainable_params']:,} trainable / "
        f"{counts['total_params']:,} total",
        flush=True,
    )
    compression_metadata = load_json(source / "compression_metadata.json")
    run_dataset = args.train_dataset if variant.trainable else "none"
    run_dir = config.root / "runs" / args.size / variant.key / run_dataset

    if variant.trainable and args.action in {"train", "train-evaluate"}:
        final = run_dir / "final" / "adapter.safetensors"
        if final.is_file():
            print(f"cached final adapter: {final}", flush=True)
            load_trainable_state(model, final.parent)
        else:
            train_adapter(
                model,
                tokenizer,
                config=config,
                variant=variant,
                size=args.size,
                train_dataset=args.train_dataset,
                run_dir=run_dir,
                compression_metadata=compression_metadata,
                resume=args.resume,
            )
    elif variant.trainable:
        load_trainable_state(model, run_dir / "final")

    if args.action in {"evaluate", "train-evaluate"}:
        result_path = run_dir / "result.json"
        evaluate_model(
            model,
            tokenizer,
            config=config,
            variant=variant,
            size=args.size,
            train_dataset=run_dataset,
            tasks=args.eval_tasks,
            output_path=result_path,
            compression_metadata=compression_metadata,
        )
        print(f"wrote {result_path}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

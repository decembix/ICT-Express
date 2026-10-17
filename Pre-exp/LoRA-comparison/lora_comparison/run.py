"""Resumable orchestrator for the complete 3-size x 8-condition study."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .protocol import EVAL_TASKS, SIZE_SPECS, TRAIN_DATASETS, VARIANTS, RunConfig, load_json, save_json
from .report import render
from .spread import build_backbones


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--phase", choices=("build", "train", "evaluate", "all", "report"), default="all")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--spread-root", default="/home/xyzysh/work/SPREAD")
    parser.add_argument("--teacher", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--teacher-revision", default="6d84c48581ece794365f2b8e9cfb043c68ade9c5")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--sizes", nargs="+", choices=tuple(SIZE_SPECS), default=list(SIZE_SPECS))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--train-datasets", nargs="+", choices=TRAIN_DATASETS, default=list(TRAIN_DATASETS))
    parser.add_argument("--eval-tasks", nargs="+", choices=EVAL_TASKS, default=list(EVAL_TASKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--wikitext-eval-blocks", type=int, default=128)
    parser.add_argument("--piqa-eval-limit", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--rerun-evaluation", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        output_root=args.output_root,
        spread_root=args.spread_root,
        teacher=args.teacher,
        teacher_revision=args.teacher_revision,
        local_files_only=args.local_files_only,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        eval_seed=args.eval_seed,
        rank=args.rank,
        alpha=args.alpha,
        train_steps=args.train_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        max_length=args.max_length,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_batch_size=args.eval_batch_size,
        wikitext_eval_blocks=args.wikitext_eval_blocks,
        piqa_eval_limit=args.piqa_eval_limit,
        gradient_checkpointing=args.gradient_checkpointing,
    ).validate()


def _run_worker(command: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    config = _config(args)
    if config.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    config.root.mkdir(parents=True, exist_ok=True)
    config_path = config.root / "experiment_config.json"
    if config_path.exists():
        existing = load_json(config_path)
        if existing != config.to_dict():
            raise RuntimeError(
                f"{config_path} belongs to a different protocol. Use another --output-root."
            )
    else:
        save_json(config_path, config.to_dict())

    if args.phase in {"build", "all"}:
        if args.dry_run:
            print(f"would build SPREAD backbones: {args.sizes}")
        else:
            build_backbones(config, args.sizes, force=args.force_build)
        if args.phase == "build":
            return

    if args.phase == "report":
        for path in render(config.root).values():
            print(path)
        return

    action = {
        "train": "train",
        "evaluate": "evaluate",
        "all": "train-evaluate",
    }[args.phase]
    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": str(config.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    failures = []
    for size in args.sizes:
        for variant_key in args.variants:
            variant = VARIANTS[variant_key]
            if args.phase == "train" and not variant.trainable:
                print(f"skip evaluation-only baseline during train phase: {size}/{variant_key}")
                continue
            datasets = list(args.train_datasets) if variant.trainable else ["none"]
            for train_dataset in datasets:
                run_dir = config.root / "runs" / size / variant_key / train_dataset
                result = run_dir / "result.json"
                if action == "train-evaluate" and result.is_file() and not args.rerun_evaluation:
                    print(f"cached result: {result}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "lora_comparison.worker",
                    "--config",
                    str(config_path),
                    "--action",
                    action,
                    "--size",
                    size,
                    "--variant",
                    variant_key,
                    "--train-dataset",
                    train_dataset,
                    "--eval-tasks",
                    *args.eval_tasks,
                    "--resume" if args.resume else "--no-resume",
                ]
                try:
                    _run_worker(command, run_dir / f"{action}.log", env, args.dry_run)
                except subprocess.CalledProcessError as error:
                    failures.append(
                        {"size": size, "variant": variant_key, "dataset": train_dataset,
                         "return_code": error.returncode}
                    )
                    if not args.continue_on_error:
                        raise
    if not args.dry_run:
        paths = render(config.root)
        for path in paths.values():
            print(path)
    if failures:
        save_json(config.root / "failed_runs.json", failures)
        raise SystemExit(f"{len(failures)} condition(s) failed; see failed_runs.json")


if __name__ == "__main__":
    main()

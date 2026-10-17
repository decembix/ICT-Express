"""Canonical experiment contract.

The code intentionally separates *compression*, *adaptation*, and *evaluation*
seeds.  The default compression uses SPREAD's committed WikiText-103
calibration sufficient statistics.  No teacher is loaded during adaptation and
no teacher-derived loss is used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TEACHER_ID = "allenai/OLMoE-1B-7B-0924"
TEACHER_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
PIQA_DATASET = "baber/piqa"
PIQA_REVISION = "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1"


@dataclass(frozen=True)
class SizeSpec:
    label: str
    target_billions: float
    num_groups: int
    selected_experts: int

    def selection_percent(self, total_experts: int) -> float:
        if not 1 <= self.selected_experts <= total_experts:
            raise ValueError(
                f"{self.label}: selected_experts={self.selected_experts} is invalid "
                f"for E={total_experts}"
            )
        if self.num_groups > self.selected_experts:
            raise ValueError(
                f"{self.label}: K={self.num_groups} exceeds selected experts "
                f"C={self.selected_experts}"
            )
        return 100.0 * self.selected_experts / total_experts


# OLMoE has about 0.47465B non-expert parameters and 0.100663B parameters per
# retained group.  These K values are the nearest integral widths to 1/3/5B.
# C/K stays approximately equal to the original SPREAD 10/8 setting.
SIZE_SPECS: dict[str, SizeSpec] = {
    "1b": SizeSpec("1b", 1.0, num_groups=8, selected_experts=10),
    "3b": SizeSpec("3b", 3.0, num_groups=25, selected_experts=32),
    "5b": SizeSpec("5b", 5.0, num_groups=45, selected_experts=57),
}


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    router_mode: str
    router_lora: bool
    expert_lora: str

    @property
    def trainable(self) -> bool:
        return self.router_lora or self.router_mode == "lowrank" or self.expert_lora != "none"


# ``expert_lora`` is either per-group (router variants), on the concatenated
# SPREAD FFN (uniform variant), or absent.
VARIANTS: dict[str, VariantSpec] = {
    "router_lora": VariantSpec(
        "router_lora", "Router+LoRA / Expert frozen", "base", True, "none"
    ),
    "router_lora_expert_lora": VariantSpec(
        "router_lora_expert_lora", "Router+LoRA / Expert+LoRA", "base", True, "group"
    ),
    "router_frozen": VariantSpec(
        "router_frozen", "Router frozen / Expert frozen", "base", False, "none"
    ),
    "router_frozen_expert_lora": VariantSpec(
        "router_frozen_expert_lora", "Router frozen / Expert+LoRA", "base", False, "group"
    ),
    "uniform": VariantSpec(
        "uniform", "No router / SPREAD uniform concat", "uniform", False, "none"
    ),
    "uniform_expert_lora": VariantSpec(
        "uniform_expert_lora", "No router / concat FFN+LoRA", "uniform", False, "concat"
    ),
    "lowrank_router": VariantSpec(
        "lowrank_router", "Low-rank router / Expert frozen", "lowrank", False, "none"
    ),
    "lowrank_router_expert_lora": VariantSpec(
        "lowrank_router_expert_lora", "Low-rank router / Expert+LoRA", "lowrank", False, "group"
    ),
}

TRAIN_DATASETS = ("wikitext", "piqa")
EVAL_TASKS = ("wikitext", "piqa")


@dataclass
class RunConfig:
    output_root: str
    spread_root: str
    teacher: str = TEACHER_ID
    teacher_revision: str = TEACHER_REVISION
    local_files_only: bool = False
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    seed: int = 42
    eval_seed: int = 42
    rank: int = 8
    alpha: float = 8.0
    train_steps: int = 100
    warmup_steps: int = 10
    learning_rate: float = 5e-5
    min_lr: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 4
    micro_batch_size: int = 1
    max_length: int = 512
    save_every: int = 25
    log_every: int = 5
    eval_batch_size: int = 4
    wikitext_eval_blocks: int | None = 128
    piqa_eval_limit: int | None = None
    gradient_checkpointing: bool = True

    def validate(self) -> "RunConfig":
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        if self.train_steps <= 0:
            raise ValueError("train_steps must be positive")
        if not 0 <= self.warmup_steps <= self.train_steps:
            raise ValueError("warmup_steps must lie in [0, train_steps]")
        if self.learning_rate <= 0 or not 0 <= self.min_lr <= self.learning_rate:
            raise ValueError("require learning_rate > 0 and 0 <= min_lr <= learning_rate")
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.batch_size % self.micro_batch_size:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        if self.max_length < 16:
            raise ValueError("max_length must be at least 16")
        if self.save_every <= 0 or self.log_every <= 0:
            raise ValueError("save_every and log_every must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        return self

    @property
    def root(self) -> Path:
        return Path(self.output_root).expanduser().resolve()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so interrupted jobs do not look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

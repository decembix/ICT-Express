"""Build size-controlled SPREAD backbones from the tracked SPREAD code.

This package does not copy private output checkpoints.  It consumes only files
that are tracked by the SPREAD repository: the HEM implementation, OLMoE model
adapter, Table-1 loader, and committed WikiText-103 sufficient statistics.
"""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .modeling import count_parameters
from .protocol import RunConfig, SIZE_SPECS, SizeSpec, canonical_json_sha256, load_json, save_json


REQUIRED_SPREAD_FILES = (
    "common/HEM/hem_core.py",
    "common/modules/expert_importance.py",
    "common/modules/expert_merger.py",
    "common/modules/gram_logdet.py",
    "common/modules/model_adapters.py",
    "common/artifacts/wikitext103/calibration_manifest.json",
    "common/artifacts/wikitext103/importance_scores.json",
    "common/artifacts/wikitext103/gram_matrices.pt.b64",
    "experiments/olmoe/table1_noKD/build_table1_models.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_spread_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    missing = [relative for relative in REQUIRED_SPREAD_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"SPREAD root {root} is missing tracked dependencies: {missing}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def spread_identity(root: Path) -> dict[str, Any]:
    commit = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    tracked = None
    try:
        listing = subprocess.run(
            ["git", "ls-files", *REQUIRED_SPREAD_FILES], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.splitlines()
        tracked = sorted(listing) == sorted(REQUIRED_SPREAD_FILES)
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "root": str(root),
        "commit": commit,
        "all_required_files_git_tracked": tracked,
        "files": {
            relative: sha256_file(root / relative) for relative in REQUIRED_SPREAD_FILES
        },
    }


def _model_kwargs(config: RunConfig) -> dict[str, Any]:
    local = Path(config.teacher).expanduser()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": config.local_files_only,
    }
    if not local.exists():
        kwargs["revision"] = config.teacher_revision
    return kwargs


def _router_initialization(
    teacher: torch.nn.Module,
    plan: dict[str, Any],
) -> torch.Tensor:
    """REAP-weighted teacher-router rows for every merged SPREAD group."""
    layers = plan["layers"]
    rows: list[torch.Tensor] = []
    for layer_index, layer in enumerate(teacher.model.layers):
        groups = layers[str(layer_index)]["groups"]
        teacher_router = layer.mlp.gate.weight.detach().float().cpu()
        group_rows = []
        for group in groups:
            members = torch.tensor(group["members"], dtype=torch.long)
            coefficients = torch.tensor(group["m_reap_weights"], dtype=torch.float32)
            coefficients = coefficients / coefficients.sum().clamp_min(1e-12)
            group_rows.append((teacher_router[members] * coefficients[:, None]).sum(0))
        rows.append(torch.stack(group_rows))
    return torch.stack(rows)


def _torch_save_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def backbone_dir(config: RunConfig, size: str) -> Path:
    return config.root / "backbones" / size


def backbone_complete(config: RunConfig, size: str) -> bool:
    destination = backbone_dir(config, size)
    required = (
        destination / "config.json",
        destination / "model.safetensors.index.json",
        destination / "spread_plan.json",
        destination / "router_init.pt",
        destination / "compression_metadata.json",
    )
    # Small checkpoints may be stored as one shard.
    model_present = (destination / "model.safetensors").is_file() or required[1].is_file()
    return model_present and all(path.is_file() for path in required if path != required[1])


@torch.no_grad()
def build_backbones(
    config: RunConfig,
    sizes: list[str],
    *,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build one or more backbones while loading the 7B teacher only once."""
    unknown = sorted(set(sizes) - set(SIZE_SPECS))
    if unknown:
        raise ValueError(f"unknown sizes {unknown}; choose from {tuple(SIZE_SPECS)}")
    root = resolve_spread_root(config.spread_root)
    identity = spread_identity(root)
    if identity["all_required_files_git_tracked"] is False:
        raise RuntimeError(
            "The selected SPREAD checkout does not track every required compression file; "
            "commit/push those files before moving this experiment to another server."
        )
    cached = {
        size: load_json(backbone_dir(config, size) / "compression_metadata.json")
        for size in sizes
        if not force and backbone_complete(config, size)
    }
    pending = [size for size in sizes if size not in cached]
    if not pending:
        return cached
    for size in pending:
        destination = backbone_dir(config, size)
        if destination.exists() and any(destination.iterdir()) and not force:
            raise RuntimeError(
                f"Incomplete backbone directory exists: {destination}. "
                "Inspect it, then rerun with --force-build to replace individual files."
            )

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.HEM.hem_core import HEMConfig, HEMMerger
    from common.modules.gram_logdet import compute_reap_scores
    from common.modules.model_adapters import OlmoeAdapter
    from experiments.olmoe.table1_noKD.build_table1_models import load_archived_statistics

    dtype = getattr(torch, config.dtype)
    kwargs = _model_kwargs(config)
    tokenizer = AutoTokenizer.from_pretrained(config.teacher, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher,
        dtype=dtype,
        device_map={"": config.device},
        low_cpu_mem_usage=True,
        **kwargs,
    ).eval()
    teacher.requires_grad_(False)
    if str(teacher.config.model_type) != "olmoe":
        raise RuntimeError(f"expected model_type='olmoe', got {teacher.config.model_type!r}")
    adapter = OlmoeAdapter()
    total_experts = int(adapter.get_num_routed_experts(teacher.config))
    collector, grams, _canonical_k8_plan = load_archived_statistics(teacher)
    conditional = collector.get_conditional_prob_scores(normalize=True)
    router_masses = collector.get_postselect_prob_scores(normalize=False)
    reap_scores = compute_reap_scores(conditional, grams)

    output: dict[str, dict[str, Any]] = dict(cached)
    for size in pending:
        spec: SizeSpec = SIZE_SPECS[size]
        destination = backbone_dir(config, size)
        destination.mkdir(parents=True, exist_ok=True)
        hem_config = HEMConfig(
            selection_percent=spec.selection_percent(total_experts),
            num_groups=spec.num_groups,
            similarity_threshold=0.95,
            svd_rank=4,
            merge_method="M-REAP-WeightAvg-Uniform",
            grouping_method="HEM-SVD",
            svd_device=config.device,
        )
        merger = HEMMerger(
            hem_config,
            router_masses=router_masses,
            ordering_scores=router_masses,
            scoring_label="reap",
            cached_plan=None,
            adapter=adapter,
        )
        print(
            f"[build {size}] SPREAD {spec.selected_experts}->{spec.num_groups} "
            f"(target {spec.target_billions:.0f}B)",
            flush=True,
        )
        started = time.time()
        student = merger.merge_experts(teacher, reap_scores, output_dir=destination)
        plan = merger.export_plan()
        if any(len(layer["groups"]) != spec.num_groups for layer in plan["layers"].values()):
            raise RuntimeError(f"{size}: HEM did not produce exactly K={spec.num_groups}")
        router_init = _router_initialization(teacher, plan)
        _torch_save_atomic(destination / "router_init.pt", router_init)
        save_json(destination / "spread_plan.json", plan)
        tokenizer.save_pretrained(destination)
        counts = count_parameters(student)
        metadata = {
            "schema_version": 1,
            "teacher": {"model": config.teacher, "revision": config.teacher_revision},
            "spread_dependency": identity,
            "calibration": {
                "source": "SPREAD committed WikiText-103 first-128 sufficient statistics",
                "seed": None,
                "manifest": "common/artifacts/wikitext103/calibration_manifest.json",
            },
            "compression": {
                "size_label": size,
                "target_billions": spec.target_billions,
                "total_teacher_experts": total_experts,
                "selected_experts": spec.selected_experts,
                "selection_percent": hem_config.selection_percent,
                "num_groups": spec.num_groups,
                "similarity_threshold": hem_config.similarity_threshold,
                "svd_rank": hem_config.svd_rank,
                "merge_method": hem_config.merge_method,
                "grouping_method": hem_config.grouping_method,
            },
            "parameters": counts,
            "actual_billions": counts["total_params"] / 1e9,
            "plan_sha256": canonical_json_sha256(plan),
            "router_initialization": "REAP-weighted mean of teacher router rows per HEM group",
            "build_seconds": time.time() - started,
        }
        save_json(destination / "compression_metadata.json", metadata)
        output[size] = metadata
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output

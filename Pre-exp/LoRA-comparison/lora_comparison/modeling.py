"""Model transformations for the eight router/LoRA conditions.

The SPREAD checkpoint contains one compatibility expert whose intermediate
dimension is ``K * I``.  Its gate/up matrices are concatenations of K merged
groups and each down-projection slice already includes SPREAD's uniform ``1/K``
output scaling.  ``RoutedGroupMLP`` exposes those same slices as K groups.  It
multiplies each down slice by K, then applies either dynamic softmax weights or
uniform ``1/K`` weights.  Consequently, uniform mode is numerically identical
to the original SPREAD concatenated FFN before any LoRA update.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .protocol import VARIANTS, VariantSpec, save_json


def _kaiming(parameter: nn.Parameter) -> None:
    nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))


class LoRAProjection(nn.Module):
    """A frozen linear projection plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        device = base.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, dtype=torch.float32, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, dtype=torch.float32, device=device)
        )
        _kaiming(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        dtype = x.dtype
        delta = F.linear(F.linear(x, self.lora_A.to(dtype)), self.lora_B.to(dtype))
        return base + delta.to(base.dtype) * self.scale


class ConcatLoRAMLP(nn.Module):
    """SPREAD's router-free concatenated Dense FFN with projection LoRA."""

    def __init__(self, dense: nn.Module, rank: int, alpha: float):
        super().__init__()
        self.gate_proj = LoRAProjection(dense.gate_proj, rank, alpha)
        self.up_proj = LoRAProjection(dense.up_proj, rank, alpha)
        self.down_proj = LoRAProjection(dense.down_proj, rank, alpha)
        self.act_fn = dense.act_fn

    def forward(self, hidden_states: torch.Tensor):
        hidden = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden), None


class UniformDenseMLP(nn.Module):
    """Unmodified SPREAD Dense FFN with the OLMoE block return signature."""

    def __init__(self, dense: nn.Module):
        super().__init__()
        self.dense = dense

    def forward(self, hidden_states: torch.Tensor):
        return self.dense(hidden_states), None


class RoutedGroupMLP(nn.Module):
    """K SPREAD groups with a frozen, LoRA-augmented, or low-rank router.

    All K groups are evaluated.  This intentionally holds FFN compute constant
    between dynamic routing and uniform concatenation; the experiment isolates
    routing weights rather than changing active capacity with sparse Top-K.
    """

    def __init__(
        self,
        dense: nn.Module,
        *,
        num_groups: int,
        router_mode: str,
        router_init: torch.Tensor | None,
        router_lora: bool,
        expert_lora: bool,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        if router_mode not in {"base", "lowrank"}:
            raise ValueError(f"unsupported routed mode {router_mode!r}")
        self.dense = dense
        self.num_groups = int(num_groups)
        self.router_mode = router_mode
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.act_fn = dense.act_fn
        device = dense.gate_proj.weight.device
        hidden_size = int(dense.gate_proj.in_features)
        total_intermediate = int(dense.gate_proj.out_features)
        if total_intermediate % self.num_groups:
            raise ValueError(
                f"intermediate width {total_intermediate} is not divisible by K={self.num_groups}"
            )
        self.group_width = total_intermediate // self.num_groups

        if router_mode == "base":
            if router_init is None or tuple(router_init.shape) != (self.num_groups, hidden_size):
                raise ValueError(
                    f"base router init must have shape {(self.num_groups, hidden_size)}"
                )
            self.router_weight = nn.Parameter(
                router_init.float().clone().to(device), requires_grad=False
            )
        else:
            self.register_parameter("router_weight", None)

        if router_lora or router_mode == "lowrank":
            self.router_lora_A = nn.Parameter(
                torch.empty(self.rank, hidden_size, dtype=torch.float32, device=device)
            )
            self.router_lora_B = nn.Parameter(
                torch.zeros(self.num_groups, self.rank, dtype=torch.float32, device=device)
            )
            _kaiming(self.router_lora_A)
        else:
            self.register_parameter("router_lora_A", None)
            self.register_parameter("router_lora_B", None)

        if expert_lora:
            k, i, h, r = self.num_groups, self.group_width, hidden_size, self.rank
            self.expert_gate_A = nn.Parameter(torch.empty(k, r, h, dtype=torch.float32, device=device))
            self.expert_gate_B = nn.Parameter(torch.zeros(k, i, r, dtype=torch.float32, device=device))
            self.expert_up_A = nn.Parameter(torch.empty(k, r, h, dtype=torch.float32, device=device))
            self.expert_up_B = nn.Parameter(torch.zeros(k, i, r, dtype=torch.float32, device=device))
            self.expert_down_A = nn.Parameter(torch.empty(k, r, i, dtype=torch.float32, device=device))
            self.expert_down_B = nn.Parameter(torch.zeros(k, h, r, dtype=torch.float32, device=device))
            for parameter in (self.expert_gate_A, self.expert_up_A, self.expert_down_A):
                _kaiming(parameter)
        else:
            for name in (
                "expert_gate_A", "expert_gate_B", "expert_up_A", "expert_up_B",
                "expert_down_A", "expert_down_B",
            ):
                self.register_parameter(name, None)

        self.last_router_entropy: float | None = None

    def _router(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_float = flat.float()
        if self.router_weight is None:
            logits = flat_float.new_zeros(flat_float.shape[0], self.num_groups)
        else:
            logits = F.linear(flat_float, self.router_weight)
        if self.router_lora_A is not None:
            delta = F.linear(
                F.linear(flat_float, self.router_lora_A), self.router_lora_B
            )
            logits = logits + delta * self.scale
        weights = F.softmax(logits, dim=-1)
        if not self.training:
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1).mean()
            self.last_router_entropy = float(entropy)
        return logits, weights

    def _input_delta(
        self,
        flat: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        dtype = flat.dtype
        compressed = torch.einsum("mh,krh->mkr", flat, A.to(dtype))
        return torch.einsum("mkr,kir->mki", compressed, B.to(dtype)) * self.scale

    def forward(self, hidden_states: torch.Tensor):
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        k, i = self.num_groups, self.group_width

        gate = self.dense.gate_proj(flat).view(-1, k, i)
        up = self.dense.up_proj(flat).view(-1, k, i)
        if self.expert_gate_A is not None:
            gate = gate + self._input_delta(flat, self.expert_gate_A, self.expert_gate_B)
            up = up + self._input_delta(flat, self.expert_up_A, self.expert_up_B)
        intermediate = self.act_fn(gate) * up

        logits, weights = self._router(flat)
        # SPREAD saved each down slice with a 1/K multiplier.  Undo it before
        # applying the new router weights; uniform weights then reproduce the
        # original checkpoint exactly.
        down = self.dense.down_proj.weight.view(original_shape[-1], k, i)
        raw_down = down.permute(1, 0, 2) * k
        output = torch.einsum(
            "mki,khi,mk->mh", intermediate, raw_down, weights.to(intermediate.dtype)
        )
        if self.expert_down_A is not None:
            dtype = intermediate.dtype
            compressed = torch.einsum(
                "mki,kri->mkr", intermediate, self.expert_down_A.to(dtype)
            )
            output = output + self.scale * torch.einsum(
                "mkr,khr,mk->mh",
                compressed,
                self.expert_down_B.to(dtype),
                weights.to(dtype),
            )
        return output.view(*original_shape), logits


def _dense_expert(layer: nn.Module) -> nn.Module:
    mlp = layer.mlp
    experts = getattr(mlp, "experts", None)
    if experts is None or len(experts) != 1:
        raise RuntimeError("Expected a SPREAD OLMoE checkpoint with one dense expert per layer")
    return experts[0]


def install_variant(
    model: nn.Module,
    variant: str | VariantSpec,
    *,
    num_groups: int,
    rank: int,
    alpha: float,
    router_init: torch.Tensor | None,
) -> VariantSpec:
    """Freeze the checkpoint and install exactly one comparison condition."""
    spec = VARIANTS[variant] if isinstance(variant, str) else variant
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = list(model.model.layers)
    if router_init is not None and tuple(router_init.shape[:2]) != (len(layers), num_groups):
        raise ValueError(
            f"router init shape {tuple(router_init.shape)} does not match "
            f"L={len(layers)}, K={num_groups}"
        )
    for layer_index, layer in enumerate(layers):
        dense = _dense_expert(layer)
        if spec.router_mode == "uniform":
            if spec.expert_lora == "concat":
                layer.mlp = ConcatLoRAMLP(dense, rank, alpha)
            elif spec.expert_lora == "none":
                layer.mlp = UniformDenseMLP(dense)
            else:
                raise ValueError(f"uniform mode does not support {spec.expert_lora!r}")
        else:
            layer.mlp = RoutedGroupMLP(
                dense,
                num_groups=num_groups,
                router_mode=spec.router_mode,
                router_init=(router_init[layer_index] if spec.router_mode == "base" else None),
                router_lora=spec.router_lora,
                expert_lora=spec.expert_lora == "group",
                rank=rank,
                alpha=alpha,
            )
    return spec


def count_parameters(model: nn.Module) -> dict[str, int]:
    unique: dict[int, nn.Parameter] = {}
    for parameter in model.parameters():
        unique.setdefault(id(parameter), parameter)
    return {
        "total_params": int(sum(parameter.numel() for parameter in unique.values())),
        "trainable_params": int(
            sum(parameter.numel() for parameter in unique.values() if parameter.requires_grad)
        ),
    }


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_trainable_state(
    model: nn.Module,
    destination: Path,
    *,
    metadata: dict[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    state = trainable_state(model)
    if not state:
        raise RuntimeError("this variant has no trainable parameters")
    save_file(state, str(destination / "adapter.safetensors"))
    save_json(destination / "adapter_metadata.json", metadata)


def load_trainable_state(model: nn.Module, source: Path) -> dict[str, Any]:
    state = load_file(str(source / "adapter.safetensors"), device="cpu")
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise RuntimeError(f"adapter mismatch: missing={missing[:5]}, extra={extra[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))
    from .protocol import load_json

    return load_json(source / "adapter_metadata.json")


def variant_metadata(spec: VariantSpec, rank: int, alpha: float) -> dict[str, Any]:
    return {"variant": asdict(spec), "rank": rank, "alpha": alpha}

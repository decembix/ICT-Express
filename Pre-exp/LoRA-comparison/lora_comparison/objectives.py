"""Teacher-free objectives shared by training and evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_choice_scores(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    continuation_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean continuation log-likelihood, shaped ``[examples, 2]`` for PIQA."""
    shifted_logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    token_logp = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    mask = continuation_mask.to(token_logp.dtype)
    scores = (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    if scores.numel() % 2:
        raise RuntimeError("PIQA scoring expects exactly two choices per example")
    return scores.view(-1, 2)


def piqa_supervised_loss(
    model,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    scores = normalized_choice_scores(
        outputs.logits, batch["input_ids"], batch["continuation_mask"]
    )
    loss = F.cross_entropy(scores, batch["labels"])
    accuracy = (scores.argmax(-1) == batch["labels"]).float().mean()
    return loss, accuracy

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DanceGRPOClipConfig:
    """
    DanceGRPO-style clipping hyperparameters.
    """

    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    eps: float = 1e-8
    # keep False to match the requested setup: no KL term for now.
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.0


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def _to_2d_advantages(
    advantages: torch.Tensor,
    target_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Convert advantages to shape [B, S].
    - [B] -> broadcast to [B, S]
    - [B, S] -> unchanged
    """
    if advantages.ndim == 1:
        if advantages.shape[0] != target_shape[0]:
            raise ValueError(
                f"advantages shape mismatch: got {tuple(advantages.shape)} for target B={target_shape[0]}"
            )
        return advantages[:, None].expand(target_shape)
    if advantages.ndim == 2:
        if tuple(advantages.shape) != target_shape:
            raise ValueError(
                f"advantages shape mismatch: got {tuple(advantages.shape)} vs target {target_shape}"
            )
        return advantages
    raise ValueError(f"advantages must be [B] or [B,S], got {tuple(advantages.shape)}")


def compute_dance_grpo_clip_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor | None = None,
    clip_range: float = 1e-4,
    adv_clip_max: float = 5.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    DanceGRPO-style objective:
      ratio = exp(new_log_probs - old_log_probs)
      advantages <- clamp(advantages, -adv_clip_max, +adv_clip_max)
      loss = mean(max(-A * ratio, -A * clip(ratio, 1-c, 1+c)))

    Args:
        new_log_probs: [B, S]
        old_log_probs: [B, S]
        advantages: [B] or [B, S]
        response_mask: [B, S] or None
    """
    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError(
            f"log_prob shape mismatch: {tuple(new_log_probs.shape)} vs {tuple(old_log_probs.shape)}"
        )
    if new_log_probs.ndim != 2:
        raise ValueError(f"log_probs must be [B,S], got {tuple(new_log_probs.shape)}")

    target_shape = (new_log_probs.shape[0], new_log_probs.shape[1])
    advantages_2d = _to_2d_advantages(advantages, target_shape).to(
        device=new_log_probs.device, dtype=torch.float32
    )
    new_log_probs = new_log_probs.float()
    old_log_probs = old_log_probs.float()

    if response_mask is not None:
        if tuple(response_mask.shape) != target_shape:
            raise ValueError(
                f"response_mask shape mismatch: {tuple(response_mask.shape)} vs {target_shape}"
            )
        response_mask = response_mask.to(device=new_log_probs.device, dtype=torch.float32)
    else:
        response_mask = torch.ones_like(new_log_probs, dtype=torch.float32)

    # Match DanceGRPO clipping on advantages.
    advantages_clipped = torch.clamp(advantages_2d, -adv_clip_max, adv_clip_max)

    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    unclipped_loss = -advantages_clipped * ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    clipped_loss = -advantages_clipped * clipped_ratio
    loss_mat = torch.maximum(unclipped_loss, clipped_loss)
    loss = _masked_mean(loss_mat, response_mask)

    # Metrics for stability/debug.
    clip_event = (torch.abs(ratio - 1.0) > clip_range).float()
    metrics = {
        "loss": loss.detach(),
        "ratio_mean": _masked_mean(ratio, response_mask).detach(),
        "ratio_std": torch.sqrt(
            _masked_mean((ratio - _masked_mean(ratio, response_mask)) ** 2, response_mask).clamp_min(eps)
        ).detach(),
        "approx_kl": (0.5 * _masked_mean((log_ratio) ** 2, response_mask)).detach(),
        "clipfrac": _masked_mean(clip_event, response_mask).detach(),
        "adv_mean": _masked_mean(advantages_clipped, response_mask).detach(),
        "adv_abs_max": advantages_clipped.abs().max().detach(),
    }
    return loss, metrics


def compute_dance_grpo_clip_loss_from_batch(
    batch: dict[str, Any],
    new_log_probs: torch.Tensor,
    config: DanceGRPOClipConfig | None = None,
    prefer_token_level_adv: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Convenience wrapper that consumes the `grpo_batch` fields from
    `collect_online_groups_as_grpo_batch(...)`.
    """
    if config is None:
        config = DanceGRPOClipConfig()

    old_log_probs = batch.get("old_log_probs", batch.get("log_probs", None))
    if old_log_probs is None:
        raise RuntimeError("batch must include `old_log_probs` or `log_probs`.")

    advantages = None
    if prefer_token_level_adv and ("advantages_t" in batch):
        advantages = batch["advantages_t"]
    elif "advantages" in batch:
        advantages = batch["advantages"]
    elif "advantages_t" in batch:
        advantages = batch["advantages_t"]
    else:
        raise RuntimeError("batch must include `advantages` or `advantages_t`.")

    response_mask = batch.get("response_mask", None)

    return compute_dance_grpo_clip_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs.to(new_log_probs.device),
        advantages=advantages.to(new_log_probs.device),
        response_mask=response_mask.to(new_log_probs.device) if response_mask is not None else None,
        clip_range=config.clip_range,
        adv_clip_max=config.adv_clip_max,
        eps=config.eps,
    )


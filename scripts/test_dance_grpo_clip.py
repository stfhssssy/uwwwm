#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.uwm.grpo_clip import (
    DanceGRPOClipConfig,
    compute_dance_grpo_clip_loss,
    compute_dance_grpo_clip_loss_from_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for Dance-style GRPO clipping.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=9)
    parser.add_argument("--clip_range", type=float, default=1e-4)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    b, s = args.batch_size, args.num_steps
    old_log_probs = torch.randn(b, s)
    new_log_probs = old_log_probs + 0.01 * torch.randn(b, s)
    advantages_scalar = torch.randn(b)
    response_mask = torch.ones(b, s)
    response_mask[:, -1] = 0.0

    loss_direct, metrics_direct = compute_dance_grpo_clip_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages_scalar,
        response_mask=response_mask,
        clip_range=args.clip_range,
        adv_clip_max=args.adv_clip_max,
    )

    batch = {
        "old_log_probs": old_log_probs,
        "advantages": advantages_scalar,
        "response_mask": response_mask,
    }
    loss_batch, metrics_batch = compute_dance_grpo_clip_loss_from_batch(
        batch,
        new_log_probs=new_log_probs,
        config=DanceGRPOClipConfig(
            clip_range=args.clip_range,
            adv_clip_max=args.adv_clip_max,
        ),
    )

    result = {
        "shape_old_log_probs": list(old_log_probs.shape),
        "shape_new_log_probs": list(new_log_probs.shape),
        "shape_advantages": list(advantages_scalar.shape),
        "shape_response_mask": list(response_mask.shape),
        "loss_direct": float(loss_direct.item()),
        "loss_batch": float(loss_batch.item()),
        "loss_abs_diff": float((loss_direct - loss_batch).abs().item()),
        "metrics_direct": {k: float(v.item()) for k, v in metrics_direct.items()},
        "metrics_batch": {k: float(v.item()) for k, v in metrics_batch.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


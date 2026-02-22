#!/usr/bin/env python3
import argparse
import os
import random
import time

import numpy as np


def _make_frame_pair(h: int = 128, w: int = 128) -> np.ndarray:
    left = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    right = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    return np.concatenate([left, right], axis=1)


def _rand_metric(center: float, scale: float) -> float:
    return float(center + scale * np.random.randn())


def main():
    parser = argparse.ArgumentParser(
        description="Log synthetic GRPO-style metrics/media to wandb for dashboard sanity check."
    )
    parser.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "uwm"))
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument("--run_name", type=str, default=f"grpo-dashboard-smoke-{int(time.time())}")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--log_media_every", type=int, default=5)
    parser.add_argument("--sleep_s", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb init mode",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as e:
        raise RuntimeError(
            "wandb is not installed. Run: python -m pip install -U wandb"
        ) from e

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        mode=args.mode,
        config={"script": "test_wandb_grpo_dashboard", "steps": args.steps},
    )

    random.seed(0)
    np.random.seed(0)

    for step in range(args.steps):
        lr = 1e-5 * (0.999 ** step)
        reward_mean = np.tanh(step / max(args.steps - 1, 1)) * 0.6 + _rand_metric(0.0, 0.02)
        reward_std = max(0.01, 0.25 - 0.2 * (step / max(args.steps - 1, 1)))
        adv_mean = _rand_metric(0.0, 0.03)
        adv_std = max(0.02, 1.0 - 0.6 * (step / max(args.steps - 1, 1)))
        old_lp = _rand_metric(-1.2, 0.08)
        new_lp = old_lp + _rand_metric(0.03, 0.03)
        ratio_mean = max(0.2, _rand_metric(1.0, 0.03))
        ratio_std = max(0.001, _rand_metric(0.08, 0.01))
        clipfrac = min(max(_rand_metric(0.15, 0.03), 0.0), 1.0)
        approx_kl = max(0.0, _rand_metric(0.01, 0.003))
        valid_ratio = min(max(_rand_metric(0.9, 0.05), 0.0), 1.0)
        loss = max(0.0, 0.4 - 0.3 * (step / max(args.steps - 1, 1)) + _rand_metric(0.0, 0.03))
        grad_norm = max(0.01, _rand_metric(0.8, 0.2))

        metrics = {
            "train/loss": loss,
            "train/lr": lr,
            "train/grad_norm": grad_norm,
            "rl/reward_mean": reward_mean,
            "rl/reward_std": reward_std,
            "rl/adv_mean": adv_mean,
            "rl/adv_std": adv_std,
            "rl/old_log_prob_mean": old_lp,
            "rl/new_log_prob_mean": new_lp,
            "rl/ratio_mean": ratio_mean,
            "rl/ratio_std": ratio_std,
            "rl/clipfrac": clipfrac,
            "rl/approx_kl": approx_kl,
            "data/valid_ratio": valid_ratio,
            "rl/ref_log_prob_mean": _rand_metric(-1.1, 0.06),
            "rl/ref_kl_mean": max(0.0, _rand_metric(0.015, 0.004)),
            "rl/kl_loss": max(0.0, _rand_metric(0.012, 0.003)),
            "rl/kl_coef": 1e-3,
            "rl/entropy_estimate": max(0.0, _rand_metric(0.9, 0.1)),
            "rl/entropy_bonus": max(0.0, _rand_metric(1e-4, 2e-5)),
            "rl/entropy_coef": 1e-4,
            # reward detail (train)
            "rl/reward_detail/reward": reward_mean,
            "rl/reward_detail/loss": max(0.0, _rand_metric(0.35, 0.03)),
            "rl/reward_detail/metrics/lpips": max(0.0, _rand_metric(0.22, 0.02)),
            "rl/reward_detail/metrics/mae": max(0.0, _rand_metric(0.11, 0.02)),
            "rl/reward_detail/metrics/mse": max(0.0, _rand_metric(0.03, 0.008)),
            "rl/reward_detail/metrics/ssim": _rand_metric(-0.5, 0.1),
            "rl/reward_detail/metrics/psnr": _rand_metric(-18.0, 2.0),
            "rl/reward_detail/reward_scalar": reward_mean,
        }

        # Validation curves
        metrics.update(
            {
                "val/loss": loss + _rand_metric(0.02, 0.01),
                "val/base_loss": loss + _rand_metric(0.01, 0.01),
                "val/reward_mean": reward_mean + _rand_metric(-0.02, 0.02),
                "val/reward_std": max(0.01, reward_std + _rand_metric(0.0, 0.02)),
                "val/adv_mean": _rand_metric(0.0, 0.03),
                "val/adv_std": max(0.02, adv_std + _rand_metric(0.0, 0.05)),
                "val/old_log_prob_mean": old_lp + _rand_metric(0.0, 0.03),
                "val/new_log_prob_mean": new_lp + _rand_metric(0.0, 0.03),
                "val/ratio_mean": max(0.2, ratio_mean + _rand_metric(0.0, 0.02)),
                "val/ratio_std": max(0.001, ratio_std + _rand_metric(0.0, 0.01)),
                "val/clipfrac": min(max(clipfrac + _rand_metric(0.0, 0.03), 0.0), 1.0),
                "val/approx_kl": max(0.0, approx_kl + _rand_metric(0.0, 0.002)),
                "val/valid_ratio": min(max(valid_ratio + _rand_metric(0.0, 0.03), 0.0), 1.0),
                "val/entropy_estimate": max(0.0, _rand_metric(0.85, 0.1)),
                "val/ref_log_prob_mean": _rand_metric(-1.05, 0.06),
                "val/ref_kl_mean": max(0.0, _rand_metric(0.014, 0.004)),
                # reward detail (val)
                "val/reward_detail/reward": reward_mean + _rand_metric(-0.02, 0.02),
                "val/reward_detail/loss": max(0.0, _rand_metric(0.37, 0.03)),
                "val/reward_detail/metrics/lpips": max(0.0, _rand_metric(0.24, 0.02)),
                "val/reward_detail/metrics/mae": max(0.0, _rand_metric(0.12, 0.02)),
                "val/reward_detail/metrics/mse": max(0.0, _rand_metric(0.035, 0.008)),
                "val/reward_detail/metrics/ssim": _rand_metric(-0.48, 0.1),
                "val/reward_detail/metrics/psnr": _rand_metric(-17.5, 2.0),
                "val/reward_detail/reward_scalar": reward_mean + _rand_metric(-0.02, 0.02),
            }
        )

        if step % args.log_media_every == 0:
            metrics["train/media/agentview_image/frame_pair"] = wandb.Image(_make_frame_pair())
            metrics["val/media/agentview_image/frame_pair"] = wandb.Image(_make_frame_pair())
            for t in range(8):
                metrics[f"train/media/agentview_image/frame_pair_t{t:02d}"] = wandb.Image(
                    _make_frame_pair(h=96, w=96)
                )
                metrics[f"val/media/agentview_image/frame_pair_t{t:02d}"] = wandb.Image(
                    _make_frame_pair(h=96, w=96)
                )

        run.log(metrics, step=step)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    run.finish()
    print("done")


if __name__ == "__main__":
    main()

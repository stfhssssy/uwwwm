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

from experiments.uwm.reward_rlvr_image import RLVRImageReward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for RLVR-compatible image reward module.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--time_steps", type=int, default=1, help="Set >1 to test multi-step aggregation.")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "last", "discount"])
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lpips", type=float, default=1.0)
    parser.add_argument("--mse", type=float, default=0.0)
    parser.add_argument("--mae", type=float, default=1.0)
    parser.add_argument("--ssim", type=float, default=0.0)
    parser.add_argument("--psnr", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    shape = (args.batch_size, 3, args.height, args.width)
    if args.time_steps > 1:
        shape = (args.batch_size, args.time_steps, 3, args.height, args.width)

    real = torch.rand(shape, device=device)
    pred = torch.rand(shape, device=device)

    reward_fn = RLVRImageReward(
        device=device,
        loss_weight={
            "lpips": args.lpips,
            "mse": args.mse,
            "mae": args.mae,
            "ssim": args.ssim,
            "psnr": args.psnr,
        },
    )
    out = reward_fn(real, pred, aggregate=args.aggregate, discount=args.discount)

    metrics = {k: float(v.mean().item()) for k, v in out["metrics"].items()}
    result = {
        "shape_real": list(real.shape),
        "shape_pred": list(pred.shape),
        "aggregate": args.aggregate,
        "discount": args.discount,
        "reward_mean": float(out["reward"].mean().item()),
        "loss_mean": float(out["loss"].mean().item()),
        "metrics_mean": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

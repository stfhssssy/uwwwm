#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import random
from datetime import datetime

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise RuntimeError(
            "This mode requires PyTorch. Please run in an environment with `torch` installed."
        ) from e
    return torch, F


def _load_episode_frames(npz_path: str, display_key: str = "image") -> np.ndarray:
    with np.load(npz_path) as data:
        if display_key not in data.files:
            raise KeyError(
                f"`{display_key}` not found in {npz_path}. Available keys: {list(data.files)}"
            )
        frames = data[display_key]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shape [T,H,W,3], got {frames.shape} from {npz_path}")
    if frames.shape[0] < 3:
        raise ValueError(f"Episode is too short (<3 frames): {npz_path}, T={frames.shape[0]}")
    return frames


def _list_episodes(rt1_dir: str):
    files = sorted(glob.glob(os.path.join(rt1_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {rt1_dir}")
    return files


def _to_bchw01(frame_hwc_uint8: np.ndarray) -> torch.Tensor:
    torch, _ = _require_torch()
    x = torch.from_numpy(frame_hwc_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return x


def _preprocess_progressor(
    frame_bchw: torch.Tensor,
    image_size: int = 84,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> torch.Tensor:
    torch, F = _require_torch()
    x = frame_bchw.clamp(0.0, 1.0).float()
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def _load_progressor_model(model_def_path: str, ckpt_path: str, device: torch.device):
    torch, _ = _require_torch()
    if not os.path.exists(model_def_path):
        raise FileNotFoundError(f"progressor model def not found: {model_def_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"progressor checkpoint not found: {ckpt_path}")

    module_name = f"progressor_simple_reward_model_test_{int(datetime.now().timestamp() * 1e6)}"
    spec = importlib.util.spec_from_file_location(module_name, model_def_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {model_def_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Model"):
        raise AttributeError(f"`Model` not found in: {model_def_path}")

    model = module.Model(model_type="resnet34", cfg=None)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _compute_progressor_reward(model, triplet_b9hw: torch.Tensor, alpha: float) -> torch.Tensor:
    torch, _ = _require_torch()
    with torch.no_grad():
        dist = model(triplet_b9hw)
    if isinstance(dist, tuple):
        dist = dist[0]
    reward = dist.mean.reshape(-1) - alpha * dist.entropy().reshape(-1)
    return reward


def _read_triplet(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    required = ["episode", "first_idx", "mid_idx", "last_idx", "display_key"]
    for k in required:
        if k not in meta:
            raise KeyError(f"Missing `{k}` in meta file: {meta_path}")
    return meta


def cmd_prepare(args):
    eps = _list_episodes(args.rt1_dir)
    rng = random.Random(args.seed)
    ep_idx = rng.randrange(len(eps))
    ep_path = eps[ep_idx]
    frames = _load_episode_frames(ep_path, args.display_key)

    t = frames.shape[0]
    first_idx = 0
    last_idx = t - 1
    mid_idx = rng.randint(1, t - 2)

    os.makedirs(os.path.dirname(os.path.abspath(args.meta_out)) or ".", exist_ok=True)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rt1_dir": os.path.abspath(args.rt1_dir),
        "episode_index_in_sorted_list": ep_idx,
        "episode": os.path.abspath(ep_path),
        "display_key": args.display_key,
        "num_frames": int(t),
        "frame_shape_hwc": [int(frames.shape[1]), int(frames.shape[2]), int(frames.shape[3])],
        "first_idx": int(first_idx),
        "mid_idx": int(mid_idx),
        "last_idx": int(last_idx),
    }
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({"mode": "prepare", "meta_out": os.path.abspath(args.meta_out), **meta}, ensure_ascii=False, indent=2))


def _build_triplet_from_meta(meta):
    frames = _load_episode_frames(meta["episode"], meta.get("display_key", "image"))
    first = frames[meta["first_idx"]]
    mid = frames[meta["mid_idx"]]
    last = frames[meta["last_idx"]]
    return first, mid, last


def cmd_case1(args):
    meta = _read_triplet(args.meta)
    _, mid, _ = _build_triplet_from_meta(meta)

    real = mid.astype(np.float32) / 255.0
    pred = mid.astype(np.float32) / 255.0  # Validation setup: both inputs use the same middle frame

    if args.rlvr_loss == "mae":
        rlvr_loss = float(np.mean(np.abs(real - pred)))
    else:
        rlvr_loss = float(np.mean((real - pred) ** 2))

    out = {
        "mode": "case1",
        "meta": os.path.abspath(args.meta),
        "episode": meta["episode"],
        "frame_indices": {
            "real_idx": meta["mid_idx"],
            "pred_idx": meta["mid_idx"],
        },
        "rlvr_loss_type": args.rlvr_loss,
        "rlvr_loss": rlvr_loss,
        "rlvr_reward": -rlvr_loss,
        "note": "progressor disabled",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_case2(args):
    torch, _ = _require_torch()
    meta = _read_triplet(args.meta)
    first, mid, last = _build_triplet_from_meta(meta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_progressor_model(args.model_def_path, args.ckpt_path, device)

    init = _preprocess_progressor(_to_bchw01(first), image_size=args.image_size)
    pred = _preprocess_progressor(_to_bchw01(mid), image_size=args.image_size)
    goal = _preprocess_progressor(_to_bchw01(last), image_size=args.image_size)
    triplet = torch.cat([init, pred, goal], dim=1).to(device)

    reward = _compute_progressor_reward(model, triplet, alpha=args.alpha).cpu()
    prog_loss = -reward

    out = {
        "mode": "case2",
        "meta": os.path.abspath(args.meta),
        "episode": meta["episode"],
        "frame_indices": {
            "init_idx": meta["first_idx"],
            "pred_idx": meta["mid_idx"],
            "goal_idx": meta["last_idx"],
        },
        "progressor_alpha": args.alpha,
        "progressor_reward_raw": float(reward.item()),
        "progressor_loss": float(prog_loss.item()),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_case3(args):
    torch, _ = _require_torch()
    meta = _read_triplet(args.meta)
    first, mid, last = _build_triplet_from_meta(meta)

    # RLVR loss branch (same setup as case1: real=mid, pred=mid)
    real = _to_bchw01(mid)
    pred_rlvr = _to_bchw01(mid)
    if args.rlvr_loss == "mae":
        rlvr_loss = torch.mean(torch.abs(real - pred_rlvr), dim=(1, 2, 3))
    else:
        rlvr_loss = torch.mean((real - pred_rlvr) ** 2, dim=(1, 2, 3))
    rlvr_reward = -rlvr_loss

    # Progressor reward branch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_progressor_model(args.model_def_path, args.ckpt_path, device)

    init = _preprocess_progressor(_to_bchw01(first), image_size=args.image_size)
    pred = _preprocess_progressor(_to_bchw01(mid), image_size=args.image_size)
    goal = _preprocess_progressor(_to_bchw01(last), image_size=args.image_size)
    triplet = torch.cat([init, pred, goal], dim=1).to(device)

    prog_reward_raw = _compute_progressor_reward(model, triplet, alpha=args.alpha).cpu()

    # Simplified alignment: identity by default; optional manual stats alignment.
    if args.prog_mu is not None and args.prog_std is not None and args.rlvr_mu is not None and args.rlvr_std is not None:
        z = (prog_reward_raw - args.prog_mu) / max(args.prog_std, args.eps)
        prog_reward_aligned = args.rlvr_mu + args.rlvr_std * z
        align_mode = "manual_stats"
    else:
        prog_reward_aligned = prog_reward_raw
        align_mode = "identity"

    w = min(max(args.weight, 0.0), 1.0)
    reward_total = (1.0 - w) * rlvr_reward + w * prog_reward_aligned
    loss_total = -reward_total

    out = {
        "mode": "case3",
        "meta": os.path.abspath(args.meta),
        "episode": meta["episode"],
        "frame_indices": {
            "init_idx": meta["first_idx"],
            "pred_idx": meta["mid_idx"],
            "goal_idx": meta["last_idx"],
            "rlvr_real_idx": meta["mid_idx"],
        },
        "rlvr_loss_type": args.rlvr_loss,
        "rlvr_loss": float(rlvr_loss.item()),
        "rlvr_reward": float(rlvr_reward.item()),
        "progressor_alpha": args.alpha,
        "progressor_reward_raw": float(prog_reward_raw.item()),
        "progressor_reward_aligned": float(prog_reward_aligned.item()),
        "align_mode": align_mode,
        "weight": w,
        "reward_total": float(reward_total.item()),
        "loss_total": float(loss_total.item()),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        description="RT1 fixed-frame validation for RLVR loss / Progressor loss / fused loss"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("prepare", help="Randomly pick one trajectory and fix first/mid/last frames")
    p.add_argument("--rt1_dir", type=str, required=True)
    p.add_argument("--display_key", type=str, default="image")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--meta_out", type=str, default="/tmp/rt1_triplet_meta.json")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("case1", help="Progressor off: compute RLVR loss only (mid vs mid)")
    p.add_argument("--meta", type=str, required=True)
    p.add_argument("--rlvr_loss", type=str, choices=["mse", "mae"], default="mse")
    p.set_defaults(func=cmd_case1)

    p = sub.add_parser("case2", help="Compute progressor loss only (first, mid, last)")
    p.add_argument("--meta", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument(
        "--model_def_path",
        type=str,
        default="/home/ssy/rlvr-progressor/progressor/Video2Reward/simple_reward_model.py",
    )
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--image_size", type=int, default=84)
    p.set_defaults(func=cmd_case2)

    p = sub.add_parser("case3", help="Compute fused RLVR + progressor loss")
    p.add_argument("--meta", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument(
        "--model_def_path",
        type=str,
        default="/home/ssy/rlvr-progressor/progressor/Video2Reward/simple_reward_model.py",
    )
    p.add_argument("--rlvr_loss", type=str, choices=["mse", "mae"], default="mse")
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--weight", type=float, default=0.2)
    p.add_argument("--image_size", type=int, default=84)

    p.add_argument("--rlvr_mu", type=float, default=None)
    p.add_argument("--rlvr_std", type=float, default=None)
    p.add_argument("--prog_mu", type=float, default=None)
    p.add_argument("--prog_std", type=float, default=None)
    p.add_argument("--eps", type=float, default=1e-6)
    p.set_defaults(func=cmd_case3)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

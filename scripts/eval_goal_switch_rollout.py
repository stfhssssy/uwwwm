#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from environments.robomimic import make_robomimic_env


def _extract_rgb_obs_keys(shape_meta: Any) -> list[str]:
    try:
        meta_obj = OmegaConf.to_container(shape_meta, resolve=True)
    except Exception:
        meta_obj = shape_meta
    if not isinstance(meta_obj, dict):
        return []
    obs_meta = meta_obj.get("obs", {})
    if not isinstance(obs_meta, dict):
        return []
    keys = []
    for k, v in obs_meta.items():
        if not isinstance(v, dict):
            continue
        if str(v.get("type", "")).lower() == "rgb":
            keys.append(str(k))
    return sorted(keys)


def _load_fixed_goal_from_hdf5(
    hdf5_path: str,
    rgb_keys: list[str],
    device: torch.device,
    demo_index: int = 0,
) -> tuple[dict[str, torch.Tensor], str]:
    if len(rgb_keys) == 0:
        raise RuntimeError("No rgb obs keys found.")
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"`data` group missing in {hdf5_path}")
        demos = sorted(list(f["data"].keys()))
        if len(demos) == 0:
            raise RuntimeError(f"No demos in {hdf5_path}")
        demo_name = demos[int(demo_index) % len(demos)]
        obs_group = f["data"][demo_name]["obs"]

        goal_obs_dict: dict[str, torch.Tensor] = {}
        for key in rgb_keys:
            if key not in obs_group:
                raise KeyError(
                    f"Key `{key}` not found in {hdf5_path}:{demo_name}/obs "
                    f"(available={list(obs_group.keys())})"
                )
            # Keep single-frame goal: [B=1, T=1, H, W, C], uint8.
            frame_hwc = np.asarray(obs_group[key][-1])
            goal_obs_dict[key] = torch.as_tensor(
                frame_hwc[None, None, ...], dtype=torch.uint8, device=device
            )
    return goal_obs_dict, demo_name


def _read_pretrain_tasks_from_ckpt(ckpt_path: str) -> list[str]:
    ckpt_dir = Path(ckpt_path).resolve().parent
    hydra_cfg = ckpt_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        return []
    cfg = OmegaConf.load(hydra_cfg)
    paths = OmegaConf.select(cfg, "dataset.hdf5_path_globs", default=[])
    if isinstance(paths, str):
        return [paths]
    if isinstance(paths, list):
        return [str(p) for p in paths]
    return []


def _pick_wrong_goal_hdf5(target_hdf5: str, candidates: list[str]) -> str:
    target_abs = str(Path(target_hdf5).resolve())
    filtered = [str(Path(p).resolve()) for p in candidates if str(Path(p).resolve()) != target_abs]
    if len(filtered) == 0:
        raise RuntimeError(
            "No alternative task found for wrong goal. Please pass --wrong_goal_hdf5 explicitly."
        )
    # Prefer a deterministic choice for reproducibility.
    return sorted(filtered)[0]


@torch.no_grad()
def run_goal_switch_rollout(
    model,
    env,
    goal_wrong: dict[str, torch.Tensor],
    goal_right: dict[str, torch.Tensor],
    num_rollouts: int,
    switch_policy_step: int,
    device: torch.device,
    seed: int = 0,
) -> dict[str, Any]:
    per_rollout = []
    successes = []
    success_before_switch = []

    for i in range(int(num_rollouts)):
        env.seed(int(seed) + i)
        obs = env.reset()
        done = False
        info = {}
        policy_step = 0

        while not done:
            obs_tensor = {k: torch.as_tensor(v, device=device)[None] for k, v in obs.items()}
            goal_obs_dict = goal_wrong if policy_step < int(switch_policy_step) else goal_right
            action = model.sample(obs_tensor, goal_obs_dict=goal_obs_dict)[0].detach().cpu().numpy()
            obs, _, done, info = env.step(action)
            policy_step += 1

        is_success = bool(info.get("success", False))
        successes.append(is_success)
        success_before_switch.append(is_success and (policy_step <= int(switch_policy_step)))
        per_rollout.append(
            {
                "rollout_id": i,
                "success": is_success,
                "policy_steps": int(policy_step),
                "switch_policy_step": int(switch_policy_step),
                "done_reason": str(info.get("done_reason", "")),
            }
        )

    out = {
        "num_rollouts": int(num_rollouts),
        "switch_policy_step": int(switch_policy_step),
        "success_rate": float(np.mean(successes)) if len(successes) > 0 else 0.0,
        "success_rate_before_switch": float(np.mean(success_before_switch)) if len(success_before_switch) > 0 else 0.0,
        "mean_policy_steps": float(np.mean([r["policy_steps"] for r in per_rollout])) if len(per_rollout) > 0 else 0.0,
        "per_rollout": per_rollout,
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Rollout eval with goal switch: wrong-goal early, right-goal later."
    )
    parser.add_argument("--ckpt_path", required=True, help="Path to pretrained model checkpoint (models.pt).")
    parser.add_argument("--target_hdf5", required=True, help="Target task dataset hdf5 (eval env + right goal source).")
    parser.add_argument("--wrong_goal_hdf5", default=None, help="Wrong-goal dataset hdf5. If not set, auto pick from ckpt's 4 tasks.")
    parser.add_argument("--switch_policy_step", type=int, default=5, help="Policy-step index to switch from wrong goal to right goal.")
    parser.add_argument("--num_rollouts", type=int, default=50, help="Number of rollouts.")
    parser.add_argument("--seed", type=int, default=0, help="Base rollout seed.")
    parser.add_argument("--demo_index", type=int, default=0, help="Demo index for extracting fixed last-frame goals.")
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--dataset_name", default="libero_black_bowl_plate", help="Dataset name passed to env factory.")
    parser.add_argument("--config_name", default="train_uwm_grpo_online.yaml", help="Config name under ./configs.")
    parser.add_argument("--max_episode_length", type=int, default=1000, help="Env max episode length.")
    parser.add_argument("--save_json", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = str((repo_root / "configs").resolve())

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name=args.config_name,
            overrides=[
                "dataset=libero_black_bowl_plate",
                f"dataset.name={args.dataset_name}",
                f"dataset.hdf5_path_globs={args.target_hdf5}",
                "model.obs_encoder.use_goal_image_cond=true",
            ],
        )
    # Do NOT call OmegaConf.resolve(cfg) here: this standalone script does not
    # initialize Hydra runtime config, and resolving `${hydra:...}` interpolations
    # would raise "HydraConfig was not set".

    model = instantiate(cfg.model).to(device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    pretrain_tasks = _read_pretrain_tasks_from_ckpt(args.ckpt_path)
    wrong_goal_hdf5 = args.wrong_goal_hdf5
    if wrong_goal_hdf5 is None:
        wrong_goal_hdf5 = _pick_wrong_goal_hdf5(args.target_hdf5, pretrain_tasks)

    rgb_keys = _extract_rgb_obs_keys(cfg.dataset.shape_meta)
    goal_right, right_demo = _load_fixed_goal_from_hdf5(
        hdf5_path=args.target_hdf5,
        rgb_keys=rgb_keys,
        device=device,
        demo_index=args.demo_index,
    )
    goal_wrong, wrong_demo = _load_fixed_goal_from_hdf5(
        hdf5_path=wrong_goal_hdf5,
        rgb_keys=rgb_keys,
        device=device,
        demo_index=args.demo_index,
    )

    env = make_robomimic_env(
        dataset_name=str(cfg.dataset.name),
        dataset_path=str(args.target_hdf5),
        shape_meta=cfg.dataset.shape_meta,
        obs_horizon=int(cfg.model.obs_encoder.num_frames),
        max_episode_length=int(args.max_episode_length),
        record=False,
        terminate_on_success=True,
    )

    try:
        result = run_goal_switch_rollout(
            model=model,
            env=env,
            goal_wrong=goal_wrong,
            goal_right=goal_right,
            num_rollouts=args.num_rollouts,
            switch_policy_step=args.switch_policy_step,
            device=device,
            seed=args.seed,
        )
    finally:
        env.close()

    output = {
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "device": str(device),
        "target_hdf5": str(Path(args.target_hdf5).resolve()),
        "wrong_goal_hdf5": str(Path(wrong_goal_hdf5).resolve()),
        "pretrain_tasks": [str(Path(p).resolve()) for p in pretrain_tasks],
        "goal_source": {
            "right_demo": right_demo,
            "wrong_demo": wrong_demo,
            "demo_index": int(args.demo_index),
            "rgb_keys": rgb_keys,
            "frame": "last",
        },
        "eval": result,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.save_json:
        out_path = Path(args.save_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

import copy
import json
import os
import random
from typing import Any

import hydra
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf

from environments.robomimic import make_robomimic_env
from experiments.utils import set_seed
from experiments.uwm.grpo_clip import (
    DanceGRPOClipConfig,
    compute_dance_grpo_clip_loss_from_batch,
)
from experiments.uwm.online_rl_data import collect_online_groups_as_grpo_batch
from experiments.uwm.reward_rlvr_image import RLVRImageReward


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    mask = mask.to(device=x.device, dtype=torch.float32)
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def _resolve_dataset_path(dataset_path_cfg: Any) -> str:
    if isinstance(dataset_path_cfg, str):
        return dataset_path_cfg
    if isinstance(dataset_path_cfg, ListConfig):
        dataset_path_cfg = list(dataset_path_cfg)
    if isinstance(dataset_path_cfg, (list, tuple)) and len(dataset_path_cfg) > 0:
        return str(dataset_path_cfg[0])
    raise RuntimeError(
        "`dataset.hdf5_path_globs` must be a string or a non-empty list of strings."
    )


def _extract_rgb_obs_keys(shape_meta: dict[str, Any]) -> list[str]:
    try:
        meta_obj = OmegaConf.to_container(shape_meta, resolve=True)
    except Exception:
        meta_obj = shape_meta

    if not isinstance(meta_obj, dict):
        return []
    obs_meta = meta_obj.get("obs", {})
    if not isinstance(obs_meta, dict):
        return []

    keys: list[str] = []
    for k, v in obs_meta.items():
        if not isinstance(v, dict):
            continue
        v_type = str(v.get("type", "")).lower()
        v_shape = v.get("shape", None)
        is_rgb = v_type == "rgb"
        if (not is_rgb) and isinstance(v_shape, (list, tuple)) and len(v_shape) >= 3:
            if int(v_shape[-1]) in (1, 3) and any(tok in str(k).lower() for tok in ("rgb", "image")):
                is_rgb = True
        if is_rgb:
            keys.append(str(k))

    if len(keys) == 0:
        keys = [str(k) for k in obs_meta.keys() if any(tok in str(k).lower() for tok in ("rgb", "image"))]
    return sorted(set(keys))


def _build_fixed_goal_obs_from_hdf5(
    dataset_path: str,
    rgb_keys: list[str],
    device: torch.device,
    demo_index: int = 0,
) -> tuple[dict[str, torch.Tensor], str]:
    """
    Build a fixed goal observation dict from one expert trajectory:
    use the last frame of each rgb view and keep T=1.
    Returned tensors follow training obs format: [B=1, T=1, H, W, C], uint8.
    """
    if len(rgb_keys) == 0:
        raise RuntimeError("No rgb keys found in shape_meta; cannot build goal image.")

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"`data` group not found in dataset: {dataset_path}")
        demos = sorted(list(f["data"].keys()))
        if len(demos) == 0:
            raise RuntimeError(f"No demos found under `data` in dataset: {dataset_path}")
        demo_name = demos[int(demo_index) % len(demos)]
        obs_group = f["data"][demo_name]["obs"]

        goal_obs_dict: dict[str, torch.Tensor] = {}
        for key in rgb_keys:
            if key not in obs_group:
                raise KeyError(
                    f"RGB key `{key}` not found in demo `{demo_name}` obs keys={list(obs_group.keys())}"
                )
            frames = obs_group[key]
            if frames.shape[0] <= 0:
                raise RuntimeError(f"Empty trajectory for key `{key}` in demo `{demo_name}`")
            last_frame_hwc = np.asarray(frames[-1])  # [H, W, C], uint8
            # Keep goal as a single frame (T=1) to match goal feature dim used by model.
            goal_np = last_frame_hwc[None, ...]  # [T=1, H, W, C]
            goal_obs_dict[key] = torch.as_tensor(goal_np, device=device, dtype=torch.uint8)[None, ...]  # [1,1,H,W,C]
    return goal_obs_dict, demo_name


def _expand_goal_obs_dict(
    goal_obs_dict: dict[str, torch.Tensor] | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if goal_obs_dict is None:
        return None
    out: dict[str, torch.Tensor] = {}
    for k, v in goal_obs_dict.items():
        t = v.to(device=device)
        if t.ndim < 1:
            raise ValueError(f"goal_obs_dict[{k}] must have batch dim, got shape={tuple(t.shape)}")
        if t.shape[0] == batch_size:
            out[k] = t
        elif t.shape[0] == 1:
            out[k] = t.expand(batch_size, *t.shape[1:])
        else:
            raise ValueError(
                f"Cannot expand goal batch for key `{k}`: goal batch={t.shape[0]}, target={batch_size}"
            )
    return out


def _build_parallel_rollout_envs(
    config,
    dataset_path: str,
    num_workers: int,
    terminate_on_success: bool,
    seed_base: int,
) -> list[Any]:
    num_workers = int(num_workers)
    if num_workers <= 1:
        return []

    envs: list[Any] = []
    for worker_id in range(num_workers):
        worker_env = make_robomimic_env(
            dataset_name=config.dataset.name,
            dataset_path=dataset_path,
            shape_meta=config.dataset.shape_meta,
            obs_horizon=config.model.obs_encoder.num_frames,
            max_episode_length=config.rollout_length,
            record=False,
            terminate_on_success=terminate_on_success,
        )
        worker_env.seed(int(seed_base) + worker_id + 1)
        worker_env.reset()
        envs.append(worker_env)
    return envs


def _stack_obs_cond(obs_cond_list: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    if len(obs_cond_list) == 0:
        raise RuntimeError("obs_cond_list is empty.")
    keys = sorted(obs_cond_list[0].keys())
    stacked = {}
    for key in keys:
        values = [sample[key] for sample in obs_cond_list]
        first = values[0]
        if torch.is_tensor(first):
            stacked[key] = torch.stack(
                [v.to(device=device) for v in values], dim=0
            )
        else:
            arr = np.asarray(values)
            stacked[key] = torch.as_tensor(arr, device=device)
    return stacked


def _unwrap_model(model):
    return getattr(model, "module", model)


def _compute_grad_norm(parameters, norm_type: float = 2.0) -> float:
    grads = []
    for p in parameters:
        if p.grad is None:
            continue
        grads.append(p.grad.detach().float().norm(norm_type))
    if len(grads) == 0:
        return 0.0
    total = torch.stack(grads).norm(norm_type)
    return float(total.item())


def _compute_kl_estimate(
    new_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_type: str = "low_var_kl",
) -> torch.Tensor:
    if kl_type == "kl":
        return new_log_probs - ref_log_probs
    if kl_type == "abs":
        return (new_log_probs - ref_log_probs).abs()
    if kl_type == "mse":
        diff = new_log_probs - ref_log_probs
        return 0.5 * diff * diff
    if kl_type == "low_var_kl":
        # Schulman low-variance estimator: exp(ref-log) - (ref-log) - 1.
        kl = ref_log_probs - new_log_probs
        ratio = torch.exp(torch.clamp(kl, min=-20.0, max=20.0))
        return ratio - kl - 1.0
    raise ValueError(f"Unsupported kl.type={kl_type}. Choose from ['low_var_kl', 'kl', 'abs', 'mse'].")


def _update_adaptive_kl_coef(
    kl_coef: float,
    observed_kl: float,
    config,
) -> float:
    """
    Adaptive KL controller (piecewise multiplicative):
      if KL > target:      coef *= factor
      elif KL < target/2:  coef /= factor
      else:                keep coef
    """
    adaptive_enable = bool(OmegaConf.select(config, "kl.adaptive.enable", default=False))
    if (not adaptive_enable) or (not np.isfinite(observed_kl)):
        return float(kl_coef)

    target = float(OmegaConf.select(config, "kl.adaptive.target", default=0.01))
    factor = float(OmegaConf.select(config, "kl.adaptive.factor", default=1.5))
    lower_ratio = float(OmegaConf.select(config, "kl.adaptive.lower_ratio", default=0.5))
    min_coef = float(OmegaConf.select(config, "kl.adaptive.min_coef", default=1e-8))
    max_coef = float(OmegaConf.select(config, "kl.adaptive.max_coef", default=10.0))

    if target <= 0.0:
        return float(np.clip(kl_coef, min_coef, max_coef))
    if factor <= 1.0:
        factor = 1.5
    lower_ratio = float(np.clip(lower_ratio, 0.0, 1.0))

    coef_new = float(kl_coef)
    if observed_kl > target:
        coef_new = coef_new * factor
    elif observed_kl < (target * lower_ratio):
        coef_new = coef_new / factor
    return float(np.clip(coef_new, min_coef, max_coef))


def _build_reward_model(config, device: torch.device):
    if config.reward.mode == "env":
        return None
    if config.reward.mode != "rlvr_image":
        raise ValueError(f"Unsupported reward.mode={config.reward.mode}.")

    loss_weight = dict(config.reward.loss_weight)
    return RLVRImageReward(
        device=device,
        loss_weight=loss_weight,
        lpips_micro_batch_size=int(config.reward.lpips_micro_batch_size),
        lpips_py_path=config.reward.lpips_py_path,
    )


def _resolve_reward_component_norm(config) -> tuple[bool, list[str] | None, dict[str, float] | None]:
    reward_cfg = config.reward
    comp_cfg = getattr(reward_cfg, "component_norm", None)
    if comp_cfg is None:
        return False, None, None

    def _cfg_get(cfg, key: str, default=None):
        if cfg is None:
            return default
        if OmegaConf.is_config(cfg):
            value = OmegaConf.select(cfg, key, default=default)
            return default if value is None else value
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    enable = bool(_cfg_get(comp_cfg, "enable", False))
    keys_cfg = _cfg_get(comp_cfg, "keys", None)
    keys: list[str] | None = None
    if keys_cfg is not None:
        if isinstance(keys_cfg, str):
            keys = [keys_cfg]
        elif callable(keys_cfg):
            keys = None
        else:
            keys = [str(k) for k in list(keys_cfg)]

    alpha_cfg = _cfg_get(comp_cfg, "alpha", None)
    alpha_from_loss_weight = bool(_cfg_get(comp_cfg, "alpha_from_loss_weight", True))

    if alpha_cfg is not None:
        if callable(alpha_cfg):
            alpha = None
        else:
            alpha = {str(k): float(v) for k, v in dict(alpha_cfg).items()}
    elif alpha_from_loss_weight:
        alpha = {str(k): float(v) for k, v in dict(reward_cfg.loss_weight).items()}
    else:
        alpha = None
    return enable, keys, alpha


def _build_ref_model(config, model, device: torch.device):
    if not bool(config.ref.enable):
        return None

    ref_model = copy.deepcopy(_unwrap_model(model)).to(device)
    ckpt_path = config.ref.checkpoint_path
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        ref_model.load_state_dict(state_dict, strict=True)
        print(f"[grpo] loaded ref checkpoint: {ckpt_path}")
    for p in ref_model.parameters():
        p.requires_grad = False
    if hasattr(ref_model, "use_logprob"):
        ref_model.use_logprob = True
    if hasattr(ref_model, "noise_scheduler") and hasattr(ref_model, "num_inference_steps"):
        ref_steps = int(getattr(ref_model, "num_inference_steps", 0) or 0)
        if ref_steps <= 0:
            ref_steps = int(config.model.num_inference_steps)
            ref_model.num_inference_steps = ref_steps
        ref_model.noise_scheduler.set_timesteps(ref_steps)
    ref_model.eval()
    return ref_model


def _maybe_init_wandb(config):
    if ("wandb" not in config) or (not bool(config.wandb.enable)):
        return None
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError(
            "wandb logging is enabled but `wandb` package is not installed."
        ) from e

    os.makedirs(config.logdir, exist_ok=True)
    run_id_path = os.path.join(config.logdir, "run_id_train_grpo_online.json")
    if bool(config.resume) and os.path.exists(run_id_path):
        with open(run_id_path, "r", encoding="utf-8") as f:
            run_id = json.load(f)["run_id"]
        resume_mode = "must"
    else:
        run_id = wandb.util.generate_id()
        with open(run_id_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id}, f)
        resume_mode = "allow" if bool(config.resume) else "never"

    project = config.wandb.project or os.environ.get("WANDB_PROJECT", "uwm")
    entity = config.wandb.entity or os.environ.get("WANDB_ENTITY", None)
    run_name = config.wandb.run_name or f"{config.exp_id}_{config.seed}"
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=config.algo,
        id=run_id,
        resume=resume_mode,
        mode=config.wandb.mode,
        config=OmegaConf.to_container(config, resolve=True),
    )
    return run


def _aggregate_reward_detail_metrics(groups: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    agg: dict[str, list[float]] = {}
    reward_total_vals: list[float] = []
    reward_part_vals: dict[str, list[float]] = {}
    adv_part_vals: dict[str, list[float]] = {}
    alpha_vals: dict[str, list[float]] = {}
    for group in groups:
        if not group.get("is_valid", False):
            continue
        group_alpha = group.get("component_alpha", None)
        if isinstance(group_alpha, dict):
            for k, v in group_alpha.items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    alpha_vals.setdefault(str(k), []).append(float(v))
        for branch in group.get("branches", []):
            r = branch.get("reward", branch.get("reward_real", None))
            if isinstance(r, (int, float)) and np.isfinite(float(r)):
                reward_total_vals.append(float(r))
            details = branch.get("reward_details", None)
            if not isinstance(details, dict):
                details = {}
            for k, v in details.items():
                if not (isinstance(v, (int, float)) and np.isfinite(float(v))):
                    continue
                v_f = float(v)
                agg.setdefault(k, []).append(v_f)
                if isinstance(k, str) and k.startswith("metrics/"):
                    comp = k.split("/", 1)[1]
                    reward_part_vals.setdefault(comp, []).append(-v_f)

            comp_adv = branch.get("advantage_components", None)
            if isinstance(comp_adv, dict):
                for k, v in comp_adv.items():
                    if isinstance(v, (int, float)) and np.isfinite(float(v)):
                        adv_part_vals.setdefault(str(k), []).append(float(v))

    out = {}
    if len(reward_total_vals) > 0:
        out[f"{prefix}/reward_total"] = float(np.mean(reward_total_vals))
    for k, vals in agg.items():
        out[f"{prefix}/reward_detail/{k}"] = float(np.mean(vals))
    for k, vals in reward_part_vals.items():
        out[f"{prefix}/reward_part/{k}"] = float(np.mean(vals))
    for k, vals in adv_part_vals.items():
        out[f"{prefix}/adv_part/{k}"] = float(np.mean(vals))
    for k, vals in alpha_vals.items():
        out[f"{prefix}/adv_alpha/{k}"] = float(np.mean(vals))
    return out


def _aggregate_group_scalar(groups: list[dict[str, Any]], key: str) -> float | None:
    vals: list[float] = []
    for group in groups:
        if not group.get("is_valid", False):
            continue
        v = group.get(key, None)
        if isinstance(v, (int, float)) and np.isfinite(float(v)):
            vals.append(float(v))
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def _is_image_like_value(x: Any) -> bool:
    try:
        t = torch.as_tensor(x, dtype=torch.float32)
    except Exception:
        return False
    if t.ndim == 3:
        return bool(t.shape[0] in (1, 3) or t.shape[-1] in (1, 3))
    if t.ndim == 4:
        return bool(t.shape[0] in (1, 3) or t.shape[1] in (1, 3) or t.shape[-1] in (1, 3))
    if t.ndim == 5:
        return _is_image_like_value(t[0])
    return False


def _to_chw01(x: Any) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim != 3:
        raise ValueError(f"Expected 3D image-like tensor, got {tuple(t.shape)}")
    if t.shape[0] in (1, 3):
        chw = t
    elif t.shape[-1] in (1, 3):
        chw = t.permute(2, 0, 1)
    else:
        raise ValueError(f"Cannot infer channels from shape {tuple(t.shape)}")
    if float(chw.max().item()) > 1.5:
        chw = chw / 255.0
    return chw.clamp(0.0, 1.0)


def _to_tchw01(x: Any) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 3:
        return _to_chw01(t).unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(f"Expected 3D/4D image-like tensor, got {tuple(t.shape)}")

    if t.shape[-1] in (1, 3):
        tchw = t.permute(0, 3, 1, 2)
    elif t.shape[1] in (1, 3):
        tchw = t
    elif t.shape[0] in (1, 3):
        tchw = t.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Cannot infer [T,C,H,W] from shape {tuple(t.shape)}")

    if float(tchw.max().item()) > 1.5:
        tchw = tchw / 255.0
    return tchw.clamp(0.0, 1.0)


def _extract_real_tchw(obs_value: Any) -> torch.Tensor:
    t = torch.as_tensor(obs_value, dtype=torch.float32)
    if t.ndim == 5:
        t = t[0]
    return _to_tchw01(t)


def _extract_pred_tchw(pred_value: Any, view_index: int = 0) -> torch.Tensor:
    t = torch.as_tensor(pred_value, dtype=torch.float32)
    if t.ndim == 6:
        t = t[0]
    if t.ndim == 5:
        t = t[view_index % t.shape[0]]
    return _to_tchw01(t)


def _align_tchw(
    real_tchw: torch.Tensor,
    pred_tchw: torch.Tensor,
    time_align: str = "tail",
    max_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    tr, tp = int(real_tchw.shape[0]), int(pred_tchw.shape[0])
    t = min(tr, tp)
    if t <= 0:
        raise RuntimeError(f"Invalid temporal length: real={tr}, pred={tp}")

    if time_align == "tail":
        real_tchw = real_tchw[-t:]
        pred_tchw = pred_tchw[-t:]
    elif time_align == "head":
        real_tchw = real_tchw[:t]
        pred_tchw = pred_tchw[:t]
    else:
        raise ValueError(f"Unsupported time_align={time_align}, choose from ['tail', 'head']")

    if max_frames > 0 and t > max_frames:
        if time_align == "tail":
            real_tchw = real_tchw[-max_frames:]
            pred_tchw = pred_tchw[-max_frames:]
        else:
            real_tchw = real_tchw[:max_frames]
            pred_tchw = pred_tchw[:max_frames]

    if tuple(pred_tchw.shape[-2:]) != tuple(real_tchw.shape[-2:]):
        pred_tchw = F.interpolate(
            pred_tchw, size=tuple(real_tchw.shape[-2:]), mode="bilinear", align_corners=False
        )
    return real_tchw, pred_tchw


def _to_hwc_uint8(chw01: torch.Tensor) -> np.ndarray:
    x = chw01.detach().cpu().clamp(0.0, 1.0)
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] > 3:
        x = x[:3]
    return (x * 255.0).round().byte().permute(1, 2, 0).numpy()


def _select_media_keys(next_obs_real: dict[str, Any], config) -> list[str]:
    if bool(config.reward.use_multi_view):
        if config.reward.obs_keys is not None:
            keys = list(config.reward.obs_keys)
        else:
            keys = sorted([k for k, v in next_obs_real.items() if _is_image_like_value(v)])
    else:
        key = config.reward.obs_key
        if key is None:
            candidates = sorted([k for k, v in next_obs_real.items() if _is_image_like_value(v)])
            key = candidates[0] if len(candidates) > 0 else None
        keys = [key] if key is not None else []
    max_views = int(config.wandb.max_media_views)
    if max_views > 0:
        keys = keys[:max_views]
    return keys


def _build_media_logs_from_groups(
    groups: list[dict[str, Any]],
    config,
    phase: str,
    step: int,
) -> dict[str, Any]:
    if ("wandb" not in config) or (not bool(config.wandb.log_media)):
        return {}
    try:
        import wandb
    except ImportError:
        return {}

    branch = None
    for group in groups:
        if not group.get("is_valid", False):
            continue
        for b in group.get("branches", []):
            if b.get("next_obs_pred", None) is None:
                continue
            if not isinstance(b.get("next_obs_real", None), dict):
                continue
            branch = b
            break
        if branch is not None:
            break
    if branch is None:
        return {}

    next_obs_real = branch["next_obs_real"]
    next_obs_pred = branch["next_obs_pred"]
    keys = _select_media_keys(next_obs_real, config)
    if len(keys) == 0:
        return {}

    media_logs = {f"{phase}/media_step": int(step)}
    max_frames = int(config.wandb.max_media_frames)
    use_multi_view = bool(config.reward.use_multi_view)
    fixed_pred_view_index = int(config.reward.pred_view_index)
    for view_index, key in enumerate(keys):
        if key not in next_obs_real:
            continue
        try:
            pred_view_index = view_index if use_multi_view else fixed_pred_view_index
            real_tchw = _extract_real_tchw(next_obs_real[key])
            pred_tchw = _extract_pred_tchw(next_obs_pred, view_index=pred_view_index)
            real_tchw, pred_tchw = _align_tchw(
                real_tchw,
                pred_tchw,
                time_align=str(config.reward.time_align),
                max_frames=max_frames,
            )
        except Exception:
            continue

        real_last = _to_hwc_uint8(real_tchw[-1])
        pred_last = _to_hwc_uint8(pred_tchw[-1])
        frame_pair = np.concatenate([real_last, pred_last], axis=1)
        media_logs[f"{phase}/media/{key}/frame_pair"] = wandb.Image(
            frame_pair,
            caption=(
                f"{phase} step={step} real_view={key} pred_view_idx={pred_view_index} "
                "frame=last"
            ),
        )

        # Per-view, per-frame comparison (real | pred), no video artifact.
        for t in range(int(real_tchw.shape[0])):
            frame_t = np.concatenate(
                [_to_hwc_uint8(real_tchw[t]), _to_hwc_uint8(pred_tchw[t])], axis=1
            )
            media_logs[f"{phase}/media/{key}/frame_pair_t{t:02d}"] = wandb.Image(
                frame_t,
                caption=(
                    f"{phase} step={step} real_view={key} pred_view_idx={pred_view_index} "
                    f"frame={t}"
                ),
            )
    return media_logs


def _sample_next_anchor(
    groups: list[dict[str, Any]],
    fallback_snapshot: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    candidates = []
    for group in groups:
        if not group.get("is_valid", False):
            continue
        for branch in group.get("branches", []):
            final_snapshot = branch.get("final_snapshot", None)
            if final_snapshot is None:
                continue
            if final_snapshot.get("sim_state", None) is None:
                continue
            candidates.append(final_snapshot)
    if len(candidates) == 0:
        return copy.deepcopy(fallback_snapshot)
    return copy.deepcopy(rng.choice(candidates))


def _save_checkpoint(
    path: str,
    step: int,
    model,
    optimizer,
    scheduler,
    scaler,
    action_normalizer=None,
    kl_coef: float | None = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "step": step,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "action_normalizer": action_normalizer,
        "kl_coef": kl_coef,
    }
    torch.save(ckpt, path)


def _maybe_resume(path: str, model, optimizer, scheduler, scaler) -> tuple[int, Any, float | None]:
    if not os.path.exists(path):
        return 0, None, None
    ckpt = torch.load(path, map_location="cpu")
    _unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return (
        int(ckpt["step"]) + 1,
        ckpt.get("action_normalizer", None),
        ckpt.get("kl_coef", None),
    )


@torch.no_grad()
def _run_validation_step(
    config,
    step: int,
    model_ref,
    ref_model,
    env,
    rollout_envs,
    initial_snapshot: dict[str, Any],
    device: torch.device,
    action_normalizer,
    reward_model,
    clip_config: DanceGRPOClipConfig,
    parallel_rollout_workers: int,
    fixed_goal_obs_dict: dict[str, torch.Tensor] | None,
    kl_coef: float | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    was_training = bool(model_ref.training)
    model_ref.eval()
    reward_component_norm, reward_component_keys, reward_component_weights = _resolve_reward_component_norm(config)
    collect_out = collect_online_groups_as_grpo_batch(
        env=env,
        model=model_ref,
        initial_snapshot=initial_snapshot,
        num_groups=int(config.validation.num_groups),
        group_size=int(config.validation.group_size),
        device=device,
        action_normalizer=action_normalizer,
        goal_obs_dict=fixed_goal_obs_dict,
        required_action_horizon=int(config.validation.required_action_horizon),
        drop_if_any_incomplete=bool(config.validation.drop_if_any_incomplete),
        terminate_on_success=bool(config.validation.terminate_on_success),
        decode_pred_obs=bool(config.validation.decode_pred_obs),
        require_logprob=True,
        compute_group_advantage=True,
        strict_dance_std=bool(config.grpo.strict_dance_std),
        rank_transform=str(config.grpo.rank_transform),
        rank_transform_components=bool(config.grpo.rank_transform_components),
        reward_fn=None,
        rlvr_reward_model=reward_model,
        reward_obs_key=config.reward.obs_key,
        reward_obs_keys=config.reward.obs_keys,
        reward_pred_view_index=int(config.reward.pred_view_index),
        reward_use_multi_view=bool(config.reward.use_multi_view),
        reward_strict_num_views=bool(config.reward.strict_num_views),
        reward_use_full_time=bool(config.reward.use_full_time),
        reward_time_align=config.reward.time_align,
        reward_pred_time_index=int(config.reward.pred_time_index),
        reward_real_time_index=int(config.reward.real_time_index),
        reward_aggregate=config.reward.aggregate,
        reward_time_aggregate=config.reward.time_aggregate,
        reward_view_aggregate=config.reward.view_aggregate,
        reward_layout_5d=config.reward.layout_5d,
        reward_layout_6d=config.reward.layout_6d,
        reward_clamp_pred=bool(config.reward.clamp_pred),
        reward_component_norm=bool(reward_component_norm),
        reward_component_keys=reward_component_keys,
        reward_component_weights=reward_component_weights,
        move_tensors_to_cpu=bool(config.validation.move_tensors_to_cpu),
        state_pool_max_size=int(config.validation.state_pool_max_size),
        seed=int(config.seed + int(config.validation.seed_offset) + step),
        batch_device=device,
        parallel_rollout_workers=int(parallel_rollout_workers),
        rollout_envs=rollout_envs,
    )
    batch = collect_out["grpo_batch"]
    if batch["action_pred"] is None:
        raise RuntimeError("validation grpo_batch.action_pred is missing.")

    obs_cond = _stack_obs_cond(batch["obs_cond"], device=device)
    action_pred = batch["action_pred"].to(device)
    timesteps = batch["timesteps"].to(device)
    latents = batch["latents"].to(device)
    next_latents = batch["next_latents"].to(device)
    goal_obs_dict_b = _expand_goal_obs_dict(
        fixed_goal_obs_dict, batch_size=int(action_pred.shape[0]), device=device
    )

    new_log_probs = model_ref.evaluate_forward_dynamics_logprob(
        obs_dict=obs_cond,
        action=action_pred,
        latents=latents,
        next_latents=next_latents,
        timesteps=timesteps,
        goal_obs_dict=goal_obs_dict_b,
    )
    base_loss, clip_metrics = compute_dance_grpo_clip_loss_from_batch(
        batch=batch,
        new_log_probs=new_log_probs,
        config=clip_config,
        prefer_token_level_adv=bool(config.grpo.prefer_token_level_adv),
    )
    loss = base_loss

    response_mask = batch.get("response_mask", None)
    if response_mask is not None:
        response_mask = response_mask.to(device=new_log_probs.device, dtype=torch.float32)

    ref_log_probs = None
    kl_mean = None
    kl_coef_used = float(config.kl.coef) if kl_coef is None else float(kl_coef)
    if ref_model is not None:
        ref_log_probs = ref_model.evaluate_forward_dynamics_logprob(
            obs_dict=obs_cond,
            action=action_pred,
            latents=latents,
            next_latents=next_latents,
            timesteps=timesteps,
            goal_obs_dict=goal_obs_dict_b,
        )
        kl_mat = _compute_kl_estimate(
            new_log_probs.float(),
            ref_log_probs.float(),
            kl_type=str(config.kl.type),
        )
        kl_mean = _masked_mean(kl_mat, response_mask)
        if bool(config.kl.enable):
            loss = loss + kl_coef_used * kl_mean

    entropy_est = -_masked_mean(new_log_probs.float(), response_mask)
    if bool(config.entropy.enable):
        loss = loss - float(config.entropy.coef) * entropy_est

    valid_ratio = float(collect_out["num_valid_groups"]) / max(
        float(collect_out["num_groups_collected"]), 1.0
    )
    val_metrics = {
        "val/loss": float(loss.item()),
        "val/base_loss": float(base_loss.item()),
        "val/reward_mean": float(batch["rewards"].mean().item()),
        "val/reward_total": float(batch["rewards"].mean().item()),
        "val/reward_std": float(batch["rewards"].std(unbiased=False).item()),
        "val/adv_mean": float(batch["advantages"].mean().item()),
        "val/adv_std": float(batch["advantages"].std(unbiased=False).item()),
        "val/old_log_prob_mean": float(batch["old_log_probs"].mean().item()),
        "val/new_log_prob_mean": float(new_log_probs.mean().item()),
        "val/ratio_mean": float(clip_metrics["ratio_mean"].item()),
        "val/ratio_std": float(clip_metrics["ratio_std"].item()),
        "val/clipfrac": float(clip_metrics["clipfrac"].item()),
        "val/approx_kl": float(clip_metrics["approx_kl"].item()),
        "val/valid_ratio": valid_ratio,
        "val/entropy_estimate": float(entropy_est.item()),
    }
    if ref_log_probs is not None:
        val_metrics["val/ref_log_prob_mean"] = float(ref_log_probs.mean().item())
    if kl_mean is not None:
        val_metrics["val/ref_kl_mean"] = float(kl_mean.item())
        val_metrics["val/kl_coef"] = float(kl_coef_used)

    val_metrics.update(_aggregate_reward_detail_metrics(collect_out["groups"], prefix="val"))
    val_std_t = _aggregate_group_scalar(collect_out["groups"], "reward_std_transformed")
    if val_std_t is not None:
        val_metrics["val/reward_std_transformed"] = val_std_t
    if was_training:
        model_ref.train()
    return val_metrics, collect_out


def train(config):
    set_seed(config.seed)

    if torch.cuda.is_available():
        device = torch.device(config.device)
    else:
        device = torch.device("cpu")
    print(f"[grpo] device={device}")

    model = instantiate(config.model).to(device)
    if config.grpo.force_enable_logprob:
        model.use_logprob = True
    if not getattr(model, "use_logprob", False):
        raise RuntimeError(
            "model.use_logprob is False. GRPO requires transition log-prob. "
            "Set model.use_logprob=True (or grpo.force_enable_logprob=True)."
        )

    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.use_amp))

    action_normalizer = None
    if config.pretrain_checkpoint_path:
        ckpt = torch.load(config.pretrain_checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        action_normalizer = ckpt.get("action_normalizer", None)
        print(f"[grpo] loaded pretrain checkpoint: {config.pretrain_checkpoint_path}")

    kl_coef = float(config.kl.coef)
    if config.resume:
        start_step, resume_action_normalizer, resume_kl_coef = _maybe_resume(
            config.train.save_path, model, optimizer, scheduler, scaler
        )
        if resume_action_normalizer is not None:
            action_normalizer = resume_action_normalizer
        if resume_kl_coef is not None:
            kl_coef = float(resume_kl_coef)
        print(f"[grpo] resumed from step={start_step}")
    else:
        start_step = 0

    dataset_path = _resolve_dataset_path(config.dataset.hdf5_path_globs)
    fixed_goal_obs_dict = None
    use_goal_image_cond = bool(getattr(config.model.obs_encoder, "use_goal_image_cond", False))
    if use_goal_image_cond:
        goal_demo_index = int(OmegaConf.select(config, "goal.demo_index", default=0))
        rgb_keys = _extract_rgb_obs_keys(config.dataset.shape_meta)
        fixed_goal_obs_dict, goal_demo_name = _build_fixed_goal_obs_from_hdf5(
            dataset_path=dataset_path,
            rgb_keys=rgb_keys,
            device=device,
            demo_index=goal_demo_index,
        )
        print(
            f"[grpo] fixed goal image enabled | source_demo={goal_demo_name} "
            f"| demo_index={goal_demo_index} | keys={rgb_keys} | frame=last | T=1"
        )
    env = make_robomimic_env(
        dataset_name=config.dataset.name,
        dataset_path=dataset_path,
        shape_meta=config.dataset.shape_meta,
        obs_horizon=config.model.obs_encoder.num_frames,
        max_episode_length=config.rollout_length,
        record=False,
        terminate_on_success=bool(config.grpo.terminate_on_success),
    )
    env.seed(config.seed)
    env.reset()
    initial_snapshot = env.get_snapshot()
    initial_snapshot_seed = copy.deepcopy(initial_snapshot)
    rollout_envs = _build_parallel_rollout_envs(
        config=config,
        dataset_path=dataset_path,
        num_workers=int(getattr(config.grpo, "parallel_rollout_workers", 1)),
        terminate_on_success=bool(config.grpo.terminate_on_success),
        seed_base=int(config.seed) + 200000,
    )
    if len(rollout_envs) > 0:
        print(f"[grpo] parallel rollout workers(train)={len(rollout_envs)}")

    val_env = None
    val_snapshot = None
    val_rollout_envs = []
    if ("validation" in config) and bool(config.validation.enable):
        val_env = make_robomimic_env(
            dataset_name=config.dataset.name,
            dataset_path=dataset_path,
            shape_meta=config.dataset.shape_meta,
            obs_horizon=config.model.obs_encoder.num_frames,
            max_episode_length=config.rollout_length,
            record=False,
            terminate_on_success=bool(config.validation.terminate_on_success),
        )
        val_env.seed(config.seed + int(config.validation.seed_offset))
        val_env.reset()
        val_snapshot = val_env.get_snapshot()
        val_rollout_envs = _build_parallel_rollout_envs(
            config=config,
            dataset_path=dataset_path,
            num_workers=int(getattr(config.validation, "parallel_rollout_workers", 1)),
            terminate_on_success=bool(config.validation.terminate_on_success),
            seed_base=int(config.seed) + int(config.validation.seed_offset) + 300000,
        )
        if len(val_rollout_envs) > 0:
            print(f"[grpo] parallel rollout workers(val)={len(val_rollout_envs)}")

    reward_model = _build_reward_model(config, device=device)
    reward_component_norm, reward_component_keys, reward_component_weights = _resolve_reward_component_norm(config)
    wandb_run = _maybe_init_wandb(config)

    clip_config = DanceGRPOClipConfig(
        clip_range=float(config.clip.clip_range),
        adv_clip_max=float(config.clip.adv_clip_max),
        eps=float(config.clip.eps),
        use_kl_loss=False,
        kl_loss_coef=0.0,
    )
    rng = random.Random(config.seed + 7)

    model_ref = _unwrap_model(model)
    ref_model = _build_ref_model(config, model_ref, device=device)
    if bool(config.kl.enable) and ref_model is None:
        raise RuntimeError("kl.enable=True requires ref.enable=True.")

    for step in range(start_step, int(config.num_steps)):
        model_ref.eval()
        try:
            collect_out = collect_online_groups_as_grpo_batch(
                env=env,
                model=model_ref,
                initial_snapshot=initial_snapshot,
                num_groups=int(config.grpo.num_groups_per_step),
                group_size=int(config.grpo.group_size),
                device=device,
                action_normalizer=action_normalizer,
                goal_obs_dict=fixed_goal_obs_dict,
                required_action_horizon=int(config.grpo.required_action_horizon),
                drop_if_any_incomplete=bool(config.grpo.drop_if_any_incomplete),
                terminate_on_success=bool(config.grpo.terminate_on_success),
                decode_pred_obs=bool(config.grpo.decode_pred_obs),
                require_logprob=True,
                compute_group_advantage=True,
                strict_dance_std=bool(config.grpo.strict_dance_std),
                rank_transform=str(config.grpo.rank_transform),
                rank_transform_components=bool(config.grpo.rank_transform_components),
                reward_fn=None,
                rlvr_reward_model=reward_model,
                reward_obs_key=config.reward.obs_key,
                reward_obs_keys=config.reward.obs_keys,
                reward_pred_view_index=int(config.reward.pred_view_index),
                reward_use_multi_view=bool(config.reward.use_multi_view),
                reward_strict_num_views=bool(config.reward.strict_num_views),
                reward_use_full_time=bool(config.reward.use_full_time),
                reward_time_align=config.reward.time_align,
                reward_pred_time_index=int(config.reward.pred_time_index),
                reward_real_time_index=int(config.reward.real_time_index),
                reward_aggregate=config.reward.aggregate,
                reward_time_aggregate=config.reward.time_aggregate,
                reward_view_aggregate=config.reward.view_aggregate,
                reward_layout_5d=config.reward.layout_5d,
                reward_layout_6d=config.reward.layout_6d,
                reward_clamp_pred=bool(config.reward.clamp_pred),
                reward_component_norm=bool(reward_component_norm),
                reward_component_keys=reward_component_keys,
                reward_component_weights=reward_component_weights,
                move_tensors_to_cpu=bool(config.grpo.move_tensors_to_cpu),
                state_pool_max_size=int(config.grpo.state_pool_max_size),
                seed=int(config.seed + step),
                batch_device=device,
                parallel_rollout_workers=int(getattr(config.grpo, "parallel_rollout_workers", 1)),
                rollout_envs=rollout_envs,
            )
        except RuntimeError as err:
            print(f"[grpo][step={step}] collect skipped: {err} | reset to initial snapshot")
            initial_snapshot = copy.deepcopy(initial_snapshot_seed)
            env.reset_to_snapshot(initial_snapshot)
            for rollout_env in rollout_envs:
                rollout_env.reset_to_snapshot(initial_snapshot)
            if val_env is not None and val_snapshot is not None:
                val_env.reset_to_snapshot(val_snapshot)
                for rollout_env in val_rollout_envs:
                    rollout_env.reset_to_snapshot(val_snapshot)
            continue

        # If any group is dropped/invalid in this collection, discard this update
        # and restart from the initial snapshot as requested.
        if int(collect_out["num_valid_groups"]) < int(collect_out["num_groups_collected"]):
            print(
                f"[grpo][step={step}] partial collect skip "
                f"({collect_out['num_valid_groups']}/{collect_out['num_groups_collected']}) "
                "-> reset to initial snapshot"
            )
            initial_snapshot = copy.deepcopy(initial_snapshot_seed)
            env.reset_to_snapshot(initial_snapshot)
            for rollout_env in rollout_envs:
                rollout_env.reset_to_snapshot(initial_snapshot)
            if val_env is not None and val_snapshot is not None:
                val_env.reset_to_snapshot(val_snapshot)
                for rollout_env in val_rollout_envs:
                    rollout_env.reset_to_snapshot(val_snapshot)
            continue

        batch = collect_out["grpo_batch"]
        if batch["action_pred"] is None:
            raise RuntimeError("grpo_batch.action_pred is missing.")

        obs_cond = _stack_obs_cond(batch["obs_cond"], device=device)
        action_pred = batch["action_pred"].to(device)
        timesteps = batch["timesteps"].to(device)
        latents = batch["latents"].to(device)
        next_latents = batch["next_latents"].to(device)
        goal_obs_dict_b = _expand_goal_obs_dict(
            fixed_goal_obs_dict, batch_size=int(action_pred.shape[0]), device=device
        )

        model_ref.train()
        optimizer.zero_grad(set_to_none=True)
        new_log_probs = model_ref.evaluate_forward_dynamics_logprob(
            obs_dict=obs_cond,
            action=action_pred,
            latents=latents,
            next_latents=next_latents,
            timesteps=timesteps,
            goal_obs_dict=goal_obs_dict_b,
        )
        loss, clip_metrics = compute_dance_grpo_clip_loss_from_batch(
            batch=batch,
            new_log_probs=new_log_probs,
            config=clip_config,
            prefer_token_level_adv=bool(config.grpo.prefer_token_level_adv),
        )

        response_mask = batch.get("response_mask", None)
        if response_mask is not None:
            response_mask = response_mask.to(device=new_log_probs.device, dtype=torch.float32)

        ref_log_probs = None
        kl_mean = None
        kl_loss = None
        if ref_model is not None:
            with torch.no_grad():
                ref_log_probs = ref_model.evaluate_forward_dynamics_logprob(
                    obs_dict=obs_cond,
                    action=action_pred,
                    latents=latents,
                    next_latents=next_latents,
                    timesteps=timesteps,
                    goal_obs_dict=goal_obs_dict_b,
                )
            kl_mat = _compute_kl_estimate(
                new_log_probs.float(),
                ref_log_probs.float(),
                kl_type=str(config.kl.type),
            )
            kl_mean = _masked_mean(kl_mat, response_mask)
            if bool(config.kl.enable):
                kl_loss = kl_mean
                loss = loss + float(kl_coef) * kl_loss

        entropy_est = -_masked_mean(new_log_probs.float(), response_mask)
        entropy_bonus = None
        if bool(config.entropy.enable):
            entropy_bonus = float(config.entropy.coef) * entropy_est
            loss = loss - entropy_bonus

        grad_norm = None
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model_ref.parameters(), float(config.clip_grad_norm)
                )
            else:
                grad_norm = _compute_grad_norm(model_ref.parameters())
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model_ref.parameters(), float(config.clip_grad_norm)
                )
            else:
                grad_norm = _compute_grad_norm(model_ref.parameters())
            optimizer.step()
        scheduler.step()
        grad_norm_value = (
            float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        )

        reward_mean = float(batch["rewards"].mean().item())
        reward_std = float(batch["rewards"].std(unbiased=False).item())
        adv_mean = float(batch["advantages"].mean().item())
        adv_std = float(batch["advantages"].std(unbiased=False).item())
        old_lp_mean = float(batch["old_log_probs"].mean().item())
        new_lp_mean = float(new_log_probs.mean().item())
        old_new_lp_diff = (new_log_probs.detach() - batch["old_log_probs"].to(device)).abs()
        old_new_lp_diff_mean = float(old_new_lp_diff.mean().item())
        old_new_lp_diff_max = float(old_new_lp_diff.max().item())
        valid_ratio = float(collect_out["num_valid_groups"]) / max(
            float(collect_out["num_groups_collected"]), 1.0
        )
        lr = scheduler.get_last_lr()[0]
        ratio_mean = float(clip_metrics["ratio_mean"].item())
        ratio_std = float(clip_metrics["ratio_std"].item())
        clipfrac = float(clip_metrics["clipfrac"].item())
        approx_kl = float(clip_metrics["approx_kl"].item())

        metrics = {
            "train/loss": float(loss.item()),
            "train/lr": float(lr),
            "train/grad_norm": float(grad_norm_value),
            "rl/reward_mean": reward_mean,
            "rl/reward_total": reward_mean,
            "rl/reward_std": reward_std,
            "rl/adv_mean": adv_mean,
            "rl/adv_std": adv_std,
            "rl/old_log_prob_mean": old_lp_mean,
            "rl/new_log_prob_mean": new_lp_mean,
            "rl/logprob_recompute_mae": old_new_lp_diff_mean,
            "rl/logprob_recompute_max": old_new_lp_diff_max,
            "rl/ratio_mean": ratio_mean,
            "rl/ratio_std": ratio_std,
            "rl/clipfrac": clipfrac,
            "rl/approx_kl": approx_kl,
            "data/valid_ratio": valid_ratio,
        }
        reward_std_t = _aggregate_group_scalar(collect_out["groups"], "reward_std_transformed")
        if reward_std_t is not None:
            metrics["rl/reward_std_transformed"] = reward_std_t
        if ref_log_probs is not None:
            metrics["rl/ref_log_prob_mean"] = float(ref_log_probs.mean().item())
        if kl_mean is not None:
            metrics["rl/ref_kl_mean"] = float(kl_mean.item())
        if kl_loss is not None:
            metrics["rl/kl_loss"] = float(kl_loss.item())
            metrics["rl/kl_coef"] = float(kl_coef)
            metrics["rl/kl_target"] = float(
                OmegaConf.select(config, "kl.adaptive.target", default=0.0)
            )
        if entropy_est is not None:
            metrics["rl/entropy_estimate"] = float(entropy_est.item())
        if entropy_bonus is not None:
            metrics["rl/entropy_bonus"] = float(entropy_bonus.item())
            metrics["rl/entropy_coef"] = float(config.entropy.coef)
        metrics.update(_aggregate_reward_detail_metrics(collect_out["groups"], prefix="rl"))

        if (
            wandb_run is not None
            and bool(config.wandb.log_media)
            and (step % int(config.wandb.media_every) == 0)
        ):
            metrics.update(
                _build_media_logs_from_groups(
                    collect_out["groups"], config, phase="train", step=step
                )
            )

        if (
            val_env is not None
            and bool(config.validation.enable)
            and int(config.validation.every) > 0
            and (step % int(config.validation.every) == 0)
        ):
            try:
                val_metrics, val_collect_out = _run_validation_step(
                    config=config,
                    step=step,
                    model_ref=model_ref,
                    ref_model=ref_model,
                    env=val_env,
                    rollout_envs=val_rollout_envs,
                    initial_snapshot=val_snapshot,
                    device=device,
                    action_normalizer=action_normalizer,
                    reward_model=reward_model,
                    clip_config=clip_config,
                    parallel_rollout_workers=int(
                        getattr(config.validation, "parallel_rollout_workers", 1)
                    ),
                    fixed_goal_obs_dict=fixed_goal_obs_dict,
                    kl_coef=float(kl_coef),
                )
                metrics.update(val_metrics)
                if (
                    wandb_run is not None
                    and bool(config.wandb.log_media)
                    and (step % int(config.wandb.val_media_every) == 0)
                ):
                    metrics.update(
                        _build_media_logs_from_groups(
                            val_collect_out["groups"], config, phase="val", step=step
                        )
                    )
            except RuntimeError as err:
                print(f"[grpo][step={step}] validation skipped: {err}")

        if wandb_run is not None and (step % int(config.wandb.log_every) == 0):
            wandb_run.log(metrics, step=step)

        if step % int(config.train.print_every) == 0:
            print(
                f"[grpo][step={step}] "
                f"loss={float(loss.item()):.6f} "
                f"reward_mean={reward_mean:.6f} reward_std={reward_std:.6f} "
                f"adv_mean={adv_mean:.6f} adv_std={adv_std:.6f} "
                f"old_lp={old_lp_mean:.6f} new_lp={new_lp_mean:.6f} "
                f"ratio_mean={ratio_mean:.6f} ratio_std={ratio_std:.6f} "
                f"clipfrac={clipfrac:.6f} approx_kl={approx_kl:.6f} "
                f"valid_ratio={valid_ratio:.3f} grad_norm={grad_norm_value:.4f} "
                f"lr={lr:.2e}"
            )

        initial_snapshot = _sample_next_anchor(
            collect_out["groups"], fallback_snapshot=initial_snapshot, rng=rng
        )

        # Update adaptive KL coefficient for next step.
        if bool(config.kl.enable) and (kl_mean is not None):
            kl_coef = _update_adaptive_kl_coef(
                kl_coef=float(kl_coef),
                observed_kl=float(kl_mean.item()),
                config=config,
            )

        if (step % int(config.train.save_every) == 0) or (step == int(config.num_steps) - 1):
            _save_checkpoint(
                path=config.train.save_path,
                step=step,
                model=model_ref,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                action_normalizer=action_normalizer,
                kl_coef=float(kl_coef),
            )

    env.close()
    for rollout_env in rollout_envs:
        rollout_env.close()
    if val_env is not None:
        val_env.close()
    for rollout_env in val_rollout_envs:
        rollout_env.close()
    if wandb_run is not None:
        wandb_run.finish()
    print("[grpo] training finished.")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_uwm_grpo_online.yaml",
)
def main(config):
    OmegaConf.resolve(config)
    train(config)


if __name__ == "__main__":
    main()

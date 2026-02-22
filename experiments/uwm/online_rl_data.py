from __future__ import annotations

import copy
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_GRPO_GROUP_SIZE = 16


def _unwrap_model(model):
    return getattr(model, "module", model)


def _to_obs_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device)[None] for k, v in obs.items()}


def _clone_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in obs.items()}


def _detach_tensor(x: Any, to_cpu: bool = True) -> Any:
    if not torch.is_tensor(x):
        return x
    y = x.detach()
    if to_cpu:
        y = y.cpu()
    return y


def _strip_batch_dim(x: Any) -> Any:
    if torch.is_tensor(x) and x.ndim > 0 and x.shape[0] == 1:
        return x[0]
    return x


def _to_tensor(x: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if torch.is_tensor(x):
        if dtype is not None and x.dtype != dtype:
            return x.to(dtype=dtype)
        return x
    return torch.as_tensor(x, dtype=dtype)


def _to_scalar(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x):
        if x.numel() == 0:
            return float(default)
        return float(x.float().mean().item())
    arr = np.asarray(x)
    if arr.size == 0:
        return float(default)
    return float(arr.astype(np.float32).mean())


def _rank_transform_values(
    values: torch.Tensor,
    mode: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Rank-transform a 1D tensor to increase group-relative separability.
    Supported modes:
      - none: identity (no transform)
      - uniform: map rank to [-1, 1]
      - gaussian: map rank quantile to N(0,1) via erfinv
    """
    if values.ndim != 1:
        raise ValueError(f"`values` must be 1D, got shape={tuple(values.shape)}")

    mode = str(mode).lower()
    if mode in ("none", ""):
        return values

    n = int(values.numel())
    if n <= 1:
        return torch.zeros_like(values, dtype=torch.float32)

    order = torch.argsort(values, stable=True)
    ranks = torch.empty(n, device=values.device, dtype=torch.float32)
    ranks[order] = torch.arange(n, device=values.device, dtype=torch.float32)
    u = (ranks + 0.5) / float(n)  # strictly in (0,1)

    if mode in ("uniform", "centered"):
        return (2.0 * u - 1.0).to(dtype=torch.float32)

    if mode in ("gaussian", "normal"):
        u = u.clamp(min=max(eps, 1e-6), max=1.0 - max(eps, 1e-6))
        return (torch.sqrt(torch.tensor(2.0, device=values.device)) * torch.erfinv(2.0 * u - 1.0)).to(
            dtype=torch.float32
        )

    raise ValueError(f"Unsupported rank_transform mode: {mode}")


def _extract_reward_value(reward_out: Any) -> float:
    """
    Accept scalar / tensor / dict result from reward_fn and return scalar reward.
    For dict outputs, preferred key order:
      reward -> score -> rewards -> loss (negated)
    """
    if isinstance(reward_out, dict):
        if "reward" in reward_out:
            return _to_scalar(reward_out["reward"])
        if "score" in reward_out:
            return _to_scalar(reward_out["score"])
        if "rewards" in reward_out:
            return _to_scalar(reward_out["rewards"])
        if "loss" in reward_out:
            return -_to_scalar(reward_out["loss"])
        raise RuntimeError("reward_fn dict output must include one of: reward/score/rewards/loss")
    return _to_scalar(reward_out)


def _extract_reward_details(reward_out: Any) -> dict[str, float]:
    """
    Extract scalar reward-related diagnostics from reward_fn output.
    Supports nested dict fields used by RLVRImageReward:
      - reward / score / rewards / loss
      - metrics: {...}
      - frame_metrics: {...}
      - view_metrics: {...}
    """
    details: dict[str, float] = {}
    if not isinstance(reward_out, dict):
        return details

    for k in ("reward", "score", "rewards", "loss"):
        if k in reward_out:
            details[k] = _to_scalar(reward_out[k])

    for parent_key in ("metrics", "frame_metrics", "view_metrics"):
        nested = reward_out.get(parent_key, None)
        if not isinstance(nested, dict):
            continue
        for k, v in nested.items():
            details[f"{parent_key}/{k}"] = _to_scalar(v)
    return details


def _infer_reward_obs_key(initial_snapshot: dict[str, Any]) -> str | None:
    obs_buffer = initial_snapshot.get("obs_buffer", None)
    if not isinstance(obs_buffer, list) or len(obs_buffer) == 0:
        return None
    first = obs_buffer[0]
    if not isinstance(first, dict) or len(first) == 0:
        return None
    # Stable key order for reproducibility. Prefer image-like observations.
    image_keys = [k for k in sorted(first.keys()) if _is_image_like_obs_value(first[k])]
    if len(image_keys) > 0:
        return image_keys[0]
    return sorted(first.keys())[0]


def _infer_reward_obs_keys(initial_snapshot: dict[str, Any]) -> list[str]:
    obs_buffer = initial_snapshot.get("obs_buffer", None)
    if not isinstance(obs_buffer, list) or len(obs_buffer) == 0:
        return []
    first = obs_buffer[0]
    if not isinstance(first, dict) or len(first) == 0:
        return []
    return [k for k in sorted(first.keys()) if _is_image_like_obs_value(first[k])]


def _is_image_like_obs_value(obs_value: Any) -> bool:
    try:
        x = _to_tensor(obs_value, dtype=torch.float32)
    except Exception:
        return False
    if x.ndim == 3:
        return bool(x.shape[0] in (1, 3) or x.shape[-1] in (1, 3))
    if x.ndim == 4:
        return bool(
            x.shape[0] in (1, 3) or x.shape[1] in (1, 3) or x.shape[-1] in (1, 3)
        )
    if x.ndim == 5:
        # [B,...], inspect the first sample.
        return _is_image_like_obs_value(x[0])
    return False


def _to_chw01(x: Any) -> torch.Tensor:
    x = _to_tensor(x, dtype=torch.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got {tuple(x.shape)}")
    if x.shape[0] in (1, 3):  # CHW
        chw = x
    elif x.shape[-1] in (1, 3):  # HWC
        chw = x.permute(2, 0, 1)
    else:
        raise ValueError(f"Cannot infer channel dimension from image shape {tuple(x.shape)}")
    if float(chw.max().item()) > 1.5:
        chw = chw / 255.0
    return chw.clamp(0.0, 1.0)


def _extract_real_bchw_from_obs(obs_value: Any, time_index: int = -1) -> torch.Tensor:
    x = _to_tensor(obs_value, dtype=torch.float32)
    if x.ndim == 3:
        return _to_chw01(x).unsqueeze(0)
    if x.ndim == 4:
        # [T,H,W,C] / [T,C,H,W] / [C,T,H,W]
        if x.shape[-1] in (1, 3) or x.shape[1] in (1, 3):
            frame = x[time_index]
            return _to_chw01(frame).unsqueeze(0)
        if x.shape[0] in (1, 3):
            frame = x[:, time_index]
            return _to_chw01(frame).unsqueeze(0)
        raise ValueError(f"Unsupported 4D real obs shape: {tuple(x.shape)}")
    if x.ndim == 5:
        # [B,...], use the first sample.
        return _extract_real_bchw_from_obs(x[0], time_index=time_index)
    raise ValueError(f"Unsupported real obs shape: {tuple(x.shape)}")


def _extract_pred_bchw(pred_value: Any, view_index: int = 0, time_index: int = -1) -> torch.Tensor:
    x = _to_tensor(pred_value, dtype=torch.float32)
    if x.ndim == 3:
        return _to_chw01(x).unsqueeze(0)
    if x.ndim == 4:
        # [C,T,H,W] / [T,C,H,W] / [T,H,W,C]
        if x.shape[0] in (1, 3):
            frame = x[:, time_index]
        else:
            frame = x[time_index]
        return _to_chw01(frame).unsqueeze(0)
    if x.ndim == 5:
        # [V,C,T,H,W] / [V,T,C,H,W]
        v = view_index % x.shape[0]
        return _extract_pred_bchw(x[v], view_index=0, time_index=time_index)
    if x.ndim == 6:
        # [B,V,C,T,H,W] / [B,V,T,C,H,W]
        return _extract_pred_bchw(x[0], view_index=view_index, time_index=time_index)
    raise ValueError(f"Unsupported predicted obs shape: {tuple(x.shape)}")


def _to_tchw01(x: Any) -> torch.Tensor:
    x = _to_tensor(x, dtype=torch.float32)
    if x.ndim == 3:
        return _to_chw01(x).unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected 3D/4D image-like tensor, got {tuple(x.shape)}")

    # [T,H,W,C]
    if x.shape[-1] in (1, 3):
        tchw = x.permute(0, 3, 1, 2)
    # [T,C,H,W]
    elif x.shape[1] in (1, 3):
        tchw = x
    # [C,T,H,W]
    elif x.shape[0] in (1, 3):
        tchw = x.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Cannot infer time/channel dims from shape {tuple(x.shape)}")

    if float(tchw.max().item()) > 1.5:
        tchw = tchw / 255.0
    return tchw.clamp(0.0, 1.0)


def _extract_real_btchw_from_obs(
    obs_value: Any,
    use_full_time: bool = True,
    time_index: int = -1,
) -> torch.Tensor:
    x = _to_tensor(obs_value, dtype=torch.float32)
    if x.ndim == 5:
        x = x[0]
    tchw = _to_tchw01(x)
    if use_full_time:
        return tchw.unsqueeze(0)  # [B,T,C,H,W]
    return tchw[time_index].unsqueeze(0)  # [B,C,H,W]


def _extract_pred_btchw(
    pred_value: Any,
    view_index: int = 0,
    use_full_time: bool = True,
    time_index: int = -1,
) -> torch.Tensor:
    x = _to_tensor(pred_value, dtype=torch.float32)
    if x.ndim == 6:
        x = x[0]
    if x.ndim == 5:
        x = x[view_index % x.shape[0]]
    tchw = _to_tchw01(x)
    if use_full_time:
        return tchw.unsqueeze(0)  # [B,T,C,H,W]
    return tchw[time_index].unsqueeze(0)  # [B,C,H,W]


def _infer_pred_num_views(pred_value: Any) -> int:
    x = _to_tensor(pred_value, dtype=torch.float32)
    if x.ndim == 6:
        return int(x.shape[1])
    if x.ndim == 5:
        # Usually [V,...] in this code path.
        return int(x.shape[0])
    return 1


def _extract_real_bvtchw_from_obs_dict(
    next_obs_real: dict[str, Any],
    obs_keys: Sequence[str],
    use_full_time: bool = True,
    time_index: int = -1,
) -> torch.Tensor:
    if len(obs_keys) == 0:
        raise RuntimeError("`obs_keys` must be non-empty for multi-view reward.")
    views = []
    for key in obs_keys:
        if key not in next_obs_real:
            raise KeyError(
                f"`{key}` not found in next_obs_real keys={list(next_obs_real.keys())}"
            )
        real_btchw = _extract_real_btchw_from_obs(
            next_obs_real[key], use_full_time=use_full_time, time_index=time_index
        )
        views.append(real_btchw)
    return torch.stack(views, dim=1)  # [B,V,T,C,H,W] or [B,V,C,H,W]


def _extract_real_bvchw_from_obs_dict(
    next_obs_real: dict[str, Any],
    obs_keys: Sequence[str],
    time_index: int = -1,
) -> torch.Tensor:
    if len(obs_keys) == 0:
        raise RuntimeError("`obs_keys` must be non-empty for multi-view reward.")
    views = []
    target_hw = None
    for key in obs_keys:
        if key not in next_obs_real:
            raise KeyError(
                f"`{key}` not found in next_obs_real keys={list(next_obs_real.keys())}"
            )
        real_bchw = _extract_real_bchw_from_obs(next_obs_real[key], time_index=time_index)
        if target_hw is None:
            target_hw = tuple(real_bchw.shape[-2:])
        elif tuple(real_bchw.shape[-2:]) != target_hw:
            real_bchw = F.interpolate(
                real_bchw, size=target_hw, mode="bilinear", align_corners=False
            )
        views.append(real_bchw)
    return torch.stack(views, dim=1)  # [B, V, C, H, W]


def _extract_pred_bvtchw(
    pred_value: Any,
    num_views: int,
    use_full_time: bool = True,
    time_index: int = -1,
    strict_num_views: bool = False,
) -> torch.Tensor:
    if num_views <= 0:
        raise RuntimeError(f"`num_views` must be positive, got {num_views}.")
    pred_views = _infer_pred_num_views(pred_value)
    if strict_num_views and pred_views != num_views:
        raise RuntimeError(
            f"Predicted view count mismatch: pred={pred_views}, expected={num_views}"
        )

    views = []
    for view_index in range(num_views):
        pred_btchw = _extract_pred_btchw(
            pred_value,
            view_index=view_index,
            use_full_time=use_full_time,
            time_index=time_index,
        )
        views.append(pred_btchw)
    return torch.stack(views, dim=1)  # [B,V,T,C,H,W] or [B,V,C,H,W]


def _extract_pred_bvchw(
    pred_value: Any,
    num_views: int,
    time_index: int = -1,
    strict_num_views: bool = False,
) -> torch.Tensor:
    if num_views <= 0:
        raise RuntimeError(f"`num_views` must be positive, got {num_views}.")
    pred_views = _infer_pred_num_views(pred_value)
    if strict_num_views and pred_views != num_views:
        raise RuntimeError(
            f"Predicted view count mismatch: pred={pred_views}, expected={num_views}"
        )

    views = []
    target_hw = None
    for view_index in range(num_views):
        pred_bchw = _extract_pred_bchw(
            pred_value, view_index=view_index, time_index=time_index
        )
        if target_hw is None:
            target_hw = tuple(pred_bchw.shape[-2:])
        elif tuple(pred_bchw.shape[-2:]) != target_hw:
            pred_bchw = F.interpolate(
                pred_bchw, size=target_hw, mode="bilinear", align_corners=False
            )
        views.append(pred_bchw)
    return torch.stack(views, dim=1)  # [B, V, C, H, W]


def _resize_image_tensor(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    if x.ndim == 4:
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    if x.ndim == 5:
        b, t = x.shape[:2]
        y = F.interpolate(
            x.reshape(b * t, *x.shape[-3:]),
            size=size_hw,
            mode="bilinear",
            align_corners=False,
        )
        return y.reshape(b, t, x.shape[-3], size_hw[0], size_hw[1])
    if x.ndim == 6:
        b, v, t = x.shape[:3]
        y = F.interpolate(
            x.reshape(b * v * t, *x.shape[-3:]),
            size=size_hw,
            mode="bilinear",
            align_corners=False,
        )
        return y.reshape(b, v, t, x.shape[-3], size_hw[0], size_hw[1])
    raise ValueError(f"Unsupported ndim for resize: {x.ndim}")


def _align_time_dim(
    real: torch.Tensor,
    pred: torch.Tensor,
    time_align: str = "tail",
) -> tuple[torch.Tensor, torch.Tensor]:
    if real.ndim == 5 and pred.ndim == 5:
        # [B,T,C,H,W]
        axis = 1
    elif real.ndim == 6 and pred.ndim == 6:
        # [B,V,T,C,H,W]
        axis = 2
    else:
        return real, pred

    tr = int(real.shape[axis])
    tp = int(pred.shape[axis])
    if tr == tp:
        return real, pred
    t = min(tr, tp)
    if t <= 0:
        raise RuntimeError(f"Invalid time length for reward alignment: real={tr}, pred={tp}")

    if time_align == "tail":
        real = real.narrow(axis, tr - t, t)
        pred = pred.narrow(axis, tp - t, t)
        return real, pred
    if time_align == "head":
        real = real.narrow(axis, 0, t)
        pred = pred.narrow(axis, 0, t)
        return real, pred
    raise ValueError(f"Unsupported time_align={time_align}, choose from ['tail', 'head']")


def make_rlvr_image_reward_fn(
    rlvr_reward_model,
    reward_obs_key: str | None,
    reward_obs_keys: Sequence[str] | None = None,
    pred_view_index: int = 0,
    use_multi_view: bool = False,
    strict_num_views: bool = False,
    use_full_time: bool = True,
    time_align: str = "tail",
    pred_time_index: int = -1,
    real_time_index: int = -1,
    aggregate: str = "mean",
    time_aggregate: str = "mean",
    view_aggregate: str = "mean",
    layout_5d: str = "btchw",
    layout_6d: str = "auto",
    clamp_pred: bool = True,
) -> Callable[[dict[str, Any]], Any]:
    """
    Build a branch-level reward_fn that uses RLVRImageReward.
    """
    if rlvr_reward_model is None:
        raise RuntimeError("`rlvr_reward_model` must not be None.")
    if (not use_multi_view) and (not reward_obs_key):
        raise RuntimeError("`reward_obs_key` must be a non-empty string.")
    if reward_obs_keys is not None:
        reward_obs_keys = list(reward_obs_keys)

    def _reward_fn(branch: dict[str, Any]) -> Any:
        next_obs_real = branch.get("next_obs_real", {})
        pred = branch.get("next_obs_pred", None)
        if pred is None:
            raise RuntimeError(
                "next_obs_pred is missing. "
                "Set decode_pred_obs=True when collecting branches."
            )

        if use_multi_view:
            obs_keys = (
                reward_obs_keys
                if reward_obs_keys is not None
                else sorted(
                    [
                        k
                        for k, v in next_obs_real.items()
                        if _is_image_like_obs_value(v)
                    ]
                )
            )
            if len(obs_keys) == 0:
                raise RuntimeError("No available obs keys for multi-view reward.")
            real = _extract_real_bvtchw_from_obs_dict(
                next_obs_real,
                obs_keys=obs_keys,
                use_full_time=use_full_time,
                time_index=real_time_index,
            )
            pred = _extract_pred_bvtchw(
                pred,
                num_views=len(obs_keys),
                use_full_time=use_full_time,
                time_index=pred_time_index,
                strict_num_views=strict_num_views,
            )
            if use_full_time:
                real, pred = _align_time_dim(real, pred, time_align=time_align)
            if tuple(pred.shape[-2:]) != tuple(real.shape[-2:]):
                pred = _resize_image_tensor(
                    pred, size_hw=(int(real.shape[-2]), int(real.shape[-1]))
                )
            if use_full_time:
                layout_6d_effective = "bvtchw" if layout_6d == "auto" else layout_6d
                return rlvr_reward_model(
                    real,
                    pred,
                    aggregate=aggregate,
                    time_aggregate=time_aggregate,
                    view_aggregate=view_aggregate,
                    layout_5d=layout_5d,
                    layout_6d=layout_6d_effective,
                    clamp_pred=clamp_pred,
                )
            return rlvr_reward_model(
                real,
                pred,
                aggregate=aggregate,
                time_aggregate=time_aggregate,
                view_aggregate=view_aggregate,
                layout_5d="bvchw",
                layout_6d=layout_6d,
                clamp_pred=clamp_pred,
            )

        if reward_obs_key not in next_obs_real:
            raise KeyError(
                f"`{reward_obs_key}` not found in next_obs_real keys={list(next_obs_real.keys())}"
            )

        real = _extract_real_btchw_from_obs(
            next_obs_real[reward_obs_key],
            use_full_time=use_full_time,
            time_index=real_time_index,
        )
        pred = _extract_pred_btchw(
            pred,
            view_index=pred_view_index,
            use_full_time=use_full_time,
            time_index=pred_time_index,
        )
        if use_full_time:
            real, pred = _align_time_dim(real, pred, time_align=time_align)
        if tuple(pred.shape[-2:]) != tuple(real.shape[-2:]):
            pred = _resize_image_tensor(
                pred, size_hw=(int(real.shape[-2]), int(real.shape[-1]))
            )

        return rlvr_reward_model(
            real,
            pred,
            aggregate=aggregate,
            time_aggregate=time_aggregate,
            view_aggregate=view_aggregate,
            layout_5d=layout_5d,
            layout_6d=layout_6d,
            clamp_pred=clamp_pred,
        )

    _reward_fn.reward_source = "rlvr_image"
    return _reward_fn


def _get_normalizer_array(action_normalizer, key: str):
    if action_normalizer is None:
        return None
    if isinstance(action_normalizer, dict):
        value = action_normalizer[key]
    else:
        value = getattr(action_normalizer, key)
    return np.asarray(value, dtype=np.float32)


def unnormalize_action_seq(
    action_seq: np.ndarray,
    action_normalizer=None,
) -> np.ndarray:
    action_seq = np.asarray(action_seq, dtype=np.float32)
    if action_seq.ndim == 1:
        action_seq = action_seq[None]
    if action_normalizer is None:
        return action_seq
    scale = _get_normalizer_array(action_normalizer, "scale")
    offset = _get_normalizer_array(action_normalizer, "offset")
    return action_seq * scale[None] + offset[None]


@torch.no_grad()
def decode_next_obs_latent(model, next_obs_latent: torch.Tensor) -> torch.Tensor:
    model = _unwrap_model(model)
    return model.obs_encoder.apply_vae(next_obs_latent, inverse=True)


@dataclass
class StatePool:
    max_size: int = 4096
    seed: int = 0

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._items: list[dict[str, Any]] = []

    def push(self, snapshot: dict[str, Any]):
        self._items.append(copy.deepcopy(snapshot))
        overflow = len(self._items) - self.max_size
        if overflow > 0:
            self._items = self._items[overflow:]

    def extend(self, snapshots: list[dict[str, Any]]):
        for snapshot in snapshots:
            self.push(snapshot)

    def sample_one(self) -> dict[str, Any]:
        if len(self._items) == 0:
            raise RuntimeError("StatePool is empty.")
        return copy.deepcopy(self._rng.choice(self._items))

    def __len__(self) -> int:
        return len(self._items)


@torch.no_grad()
def sample_branch_prediction(
    model,
    obs_tensor: dict[str, torch.Tensor],
    goal_obs_dict: dict[str, torch.Tensor] | None = None,
    decode_pred_obs: bool = True,
    require_logprob: bool = True,
) -> dict[str, Any]:
    """
    Predict one branch:
      obs(2) -> action(16) and next_obs latent.
    For GRPO training, this function should return diffusion transition
    trajectory fields: timesteps, latents, next_latents, log_probs.
    """
    model = _unwrap_model(model)

    # 1) action branch
    if hasattr(model, "sample_marginal_action"):
        action_pred = model.sample_marginal_action(obs_tensor, goal_obs_dict=goal_obs_dict)
    elif hasattr(model, "sample_joint"):
        next_obs_pred_latent, action_pred = model.sample_joint(
            obs_tensor, goal_obs_dict=goal_obs_dict
        )
        dyn_out = {"sample": next_obs_pred_latent}
    else:
        raise RuntimeError("Model does not provide action sampling interface.")

    # 2) forward dynamics branch (with optional logprob trajectory)
    if "dyn_out" not in locals():
        if hasattr(model, "sample_forward_dynamics_for_rl"):
            dyn_out = model.sample_forward_dynamics_for_rl(
                obs_tensor, action_pred, goal_obs_dict=goal_obs_dict
            )
        elif hasattr(model, "sample_forward_dynamics_with_logprob"):
            dyn_out = model.sample_forward_dynamics_with_logprob(
                obs_tensor, action_pred, goal_obs_dict=goal_obs_dict
            )
        elif hasattr(model, "sample_forward_dynamics"):
            dyn_out = {
                "sample": model.sample_forward_dynamics(
                    obs_tensor, action_pred, goal_obs_dict=goal_obs_dict
                )
            }
        else:
            raise RuntimeError("Model does not provide forward dynamics sampling interface.")

    next_obs_pred_latent = dyn_out["sample"]
    result = {"action_pred": action_pred, "next_obs_pred_latent": next_obs_pred_latent}

    if "timesteps" in dyn_out:
        result["timesteps"] = dyn_out["timesteps"]
    if "latents" in dyn_out:
        result["latents"] = dyn_out["latents"]
    if "next_latents" in dyn_out:
        result["next_latents"] = dyn_out["next_latents"]
    if "log_probs" in dyn_out:
        result["log_probs"] = dyn_out["log_probs"]
        result["old_log_probs"] = dyn_out["log_probs"]

    if require_logprob and "log_probs" not in result:
        raise RuntimeError(
            "Log-prob trajectory is required for GRPO but missing. "
            "Set model.use_logprob=True and use *_with_logprob path."
        )

    if decode_pred_obs:
        result["next_obs_pred"] = decode_next_obs_latent(model, next_obs_pred_latent)
    return result


@torch.no_grad()
def collect_branch_from_anchor(
    env,
    model,
    anchor_snapshot: dict[str, Any],
    device: torch.device,
    action_normalizer=None,
    goal_obs_dict: dict[str, torch.Tensor] | None = None,
    terminate_on_success: bool = False,
    capture_state_trace: bool = False,
    capture_obs_trace: bool = False,
    decode_pred_obs: bool = True,
    require_logprob: bool = True,
    group_id: int | None = None,
    branch_id: int | None = None,
    move_tensors_to_cpu: bool = True,
    reward_fn: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    obs = env.reset_to_snapshot(anchor_snapshot)
    obs_tensor = _to_obs_tensor(obs, device=device)
    pred = sample_branch_prediction(
        model=model,
        obs_tensor=obs_tensor,
        goal_obs_dict=goal_obs_dict,
        decode_pred_obs=decode_pred_obs,
        require_logprob=require_logprob,
    )

    action_pred_store = _strip_batch_dim(_detach_tensor(pred["action_pred"], to_cpu=move_tensors_to_cpu))
    action_pred_norm = np.asarray(_detach_tensor(pred["action_pred"], to_cpu=True))
    if action_pred_norm.ndim >= 3 and action_pred_norm.shape[0] == 1:
        action_pred_norm = action_pred_norm[0]
    action_exec = unnormalize_action_seq(action_pred_norm, action_normalizer=action_normalizer)
    rollout = env.step_with_trace(
        actions=action_exec,
        terminate_on_success=terminate_on_success,
        capture_state_trace=capture_state_trace,
        capture_obs_trace=capture_obs_trace,
        return_final_snapshot=True,
    )

    timesteps = pred.get("timesteps", None)
    latents = pred.get("latents", None)
    next_latents = pred.get("next_latents", None)
    old_log_probs = pred.get("old_log_probs", pred.get("log_probs", None))

    timesteps = _strip_batch_dim(_detach_tensor(timesteps, to_cpu=move_tensors_to_cpu))
    latents = _strip_batch_dim(_detach_tensor(latents, to_cpu=move_tensors_to_cpu))
    next_latents = _strip_batch_dim(_detach_tensor(next_latents, to_cpu=move_tensors_to_cpu))
    old_log_probs = _strip_batch_dim(_detach_tensor(old_log_probs, to_cpu=move_tensors_to_cpu))

    branch_record = {
        "group_id": group_id,
        "branch_id": branch_id,
        "anchor_snapshot": copy.deepcopy(anchor_snapshot),
        "obs_cond": _clone_obs(obs),
        "action_pred": action_pred_store,
        "timesteps": timesteps,
        "latents": latents,
        "next_latents": next_latents,
        "old_log_probs": old_log_probs,
        "next_obs_pred_latent": _strip_batch_dim(
            _detach_tensor(pred["next_obs_pred_latent"], to_cpu=move_tensors_to_cpu)
        ),
        "next_obs_pred": _strip_batch_dim(
            _detach_tensor(pred.get("next_obs_pred", None), to_cpu=move_tensors_to_cpu)
        ),
        "next_obs_real": _clone_obs(rollout["obs"]),
        "reward_real": float(rollout["reward"]),
        "reward": float(rollout["reward"]),
        "done": bool(rollout["done"]),
        "done_reason": rollout["done_reason"],
        "steps_executed": int(rollout["steps_executed"]),
        "info": copy.deepcopy(rollout["info"]),
        "final_snapshot": rollout["final_snapshot"],
        "state_trace": rollout.get("state_trace", None),
        "obs_trace": rollout.get("obs_trace", None),
    }
    if reward_fn is not None:
        reward_out = reward_fn(
            {
                "group_id": group_id,
                "branch_id": branch_id,
                "obs_cond": branch_record["obs_cond"],
                "next_obs_pred": branch_record["next_obs_pred"],
                "next_obs_pred_latent": branch_record["next_obs_pred_latent"],
                "next_obs_real": branch_record["next_obs_real"],
                "action_pred": branch_record["action_pred"],
                "reward_real": branch_record["reward_real"],
                "done": branch_record["done"],
                "steps_executed": branch_record["steps_executed"],
                "done_reason": branch_record["done_reason"],
            }
        )
        branch_record["reward"] = _extract_reward_value(reward_out)
        reward_details = _extract_reward_details(reward_out)
        reward_details["reward_scalar"] = float(branch_record["reward"])
        branch_record["reward_details"] = reward_details
        branch_record["reward_source"] = getattr(reward_fn, "reward_source", "custom_reward_fn")
    else:
        branch_record["reward_details"] = {
            "reward_real": float(branch_record["reward_real"]),
            "reward_scalar": float(branch_record["reward"]),
        }
        branch_record["reward_source"] = "env"
    return branch_record


def _prepare_branch_prediction_from_obs(
    model,
    obs_tensor: dict[str, torch.Tensor],
    action_normalizer=None,
    goal_obs_dict: dict[str, torch.Tensor] | None = None,
    decode_pred_obs: bool = True,
    require_logprob: bool = True,
    move_tensors_to_cpu: bool = True,
) -> dict[str, Any]:
    pred = sample_branch_prediction(
        model=model,
        obs_tensor=obs_tensor,
        goal_obs_dict=goal_obs_dict,
        decode_pred_obs=decode_pred_obs,
        require_logprob=require_logprob,
    )

    action_pred_store = _strip_batch_dim(_detach_tensor(pred["action_pred"], to_cpu=move_tensors_to_cpu))
    action_pred_norm = np.asarray(_detach_tensor(pred["action_pred"], to_cpu=True))
    if action_pred_norm.ndim >= 3 and action_pred_norm.shape[0] == 1:
        action_pred_norm = action_pred_norm[0]
    action_exec = unnormalize_action_seq(action_pred_norm, action_normalizer=action_normalizer)

    return {
        "action_pred": action_pred_store,
        "action_exec": action_exec,
        "timesteps": _strip_batch_dim(
            _detach_tensor(pred.get("timesteps", None), to_cpu=move_tensors_to_cpu)
        ),
        "latents": _strip_batch_dim(_detach_tensor(pred.get("latents", None), to_cpu=move_tensors_to_cpu)),
        "next_latents": _strip_batch_dim(
            _detach_tensor(pred.get("next_latents", None), to_cpu=move_tensors_to_cpu)
        ),
        "old_log_probs": _strip_batch_dim(
            _detach_tensor(pred.get("old_log_probs", pred.get("log_probs", None)), to_cpu=move_tensors_to_cpu)
        ),
        "next_obs_pred_latent": _strip_batch_dim(
            _detach_tensor(pred["next_obs_pred_latent"], to_cpu=move_tensors_to_cpu)
        ),
        "next_obs_pred": _strip_batch_dim(
            _detach_tensor(pred.get("next_obs_pred", None), to_cpu=move_tensors_to_cpu)
        ),
    }


def _rollout_from_anchor(
    rollout_env,
    anchor_snapshot: dict[str, Any],
    action_exec: np.ndarray,
    terminate_on_success: bool,
) -> dict[str, Any]:
    rollout_env.reset_to_snapshot(copy.deepcopy(anchor_snapshot))
    return rollout_env.step_with_trace(
        actions=action_exec,
        terminate_on_success=terminate_on_success,
        capture_state_trace=False,
        capture_obs_trace=False,
        return_final_snapshot=True,
    )


def _build_branch_record_from_prediction_and_rollout(
    anchor_snapshot: dict[str, Any],
    obs_cond: dict[str, np.ndarray],
    prediction: dict[str, Any],
    rollout: dict[str, Any],
    reward_fn: Callable[[dict[str, Any]], Any] | None,
    group_id: int | None,
    branch_id: int | None,
) -> dict[str, Any]:
    branch_record = {
        "group_id": group_id,
        "branch_id": branch_id,
        "anchor_snapshot": copy.deepcopy(anchor_snapshot),
        "obs_cond": _clone_obs(obs_cond),
        "action_pred": prediction["action_pred"],
        "timesteps": prediction["timesteps"],
        "latents": prediction["latents"],
        "next_latents": prediction["next_latents"],
        "old_log_probs": prediction["old_log_probs"],
        "next_obs_pred_latent": prediction["next_obs_pred_latent"],
        "next_obs_pred": prediction["next_obs_pred"],
        "next_obs_real": _clone_obs(rollout["obs"]),
        "reward_real": float(rollout["reward"]),
        "reward": float(rollout["reward"]),
        "done": bool(rollout["done"]),
        "done_reason": rollout["done_reason"],
        "steps_executed": int(rollout["steps_executed"]),
        "info": copy.deepcopy(rollout["info"]),
        "final_snapshot": rollout["final_snapshot"],
        "state_trace": rollout.get("state_trace", None),
        "obs_trace": rollout.get("obs_trace", None),
    }
    if reward_fn is not None:
        reward_out = reward_fn(
            {
                "group_id": group_id,
                "branch_id": branch_id,
                "obs_cond": branch_record["obs_cond"],
                "next_obs_pred": branch_record["next_obs_pred"],
                "next_obs_pred_latent": branch_record["next_obs_pred_latent"],
                "next_obs_real": branch_record["next_obs_real"],
                "action_pred": branch_record["action_pred"],
                "reward_real": branch_record["reward_real"],
                "done": branch_record["done"],
                "steps_executed": branch_record["steps_executed"],
                "done_reason": branch_record["done_reason"],
            }
        )
        branch_record["reward"] = _extract_reward_value(reward_out)
        reward_details = _extract_reward_details(reward_out)
        reward_details["reward_scalar"] = float(branch_record["reward"])
        branch_record["reward_details"] = reward_details
        branch_record["reward_source"] = getattr(reward_fn, "reward_source", "custom_reward_fn")
    else:
        branch_record["reward_details"] = {
            "reward_real": float(branch_record["reward_real"]),
            "reward_scalar": float(branch_record["reward"]),
        }
        branch_record["reward_source"] = "env"
    return branch_record


def validate_group_complete(
    branches: list[dict[str, Any]],
    required_action_horizon: int,
) -> tuple[bool, str | None]:
    for i, branch in enumerate(branches):
        if branch["steps_executed"] < required_action_horizon:
            return False, f"branch_{i}_incomplete_{branch['steps_executed']}/{required_action_horizon}"
    return True, None


@torch.no_grad()
def collect_group_from_anchor(
    env,
    model,
    anchor_snapshot: dict[str, Any],
    group_size: int = DEFAULT_GRPO_GROUP_SIZE,
    device: torch.device | None = None,
    action_normalizer=None,
    goal_obs_dict: dict[str, torch.Tensor] | None = None,
    required_action_horizon: int = 16,
    drop_if_any_incomplete: bool = True,
    terminate_on_success: bool = False,
    decode_pred_obs: bool = True,
    require_logprob: bool = True,
    group_id: int | None = None,
    compute_group_advantage: bool = True,
    strict_dance_std: bool = True,
    rank_transform: str = "none",
    rank_transform_components: bool = True,
    reward_component_norm: bool = False,
    reward_component_keys: Sequence[str] | None = None,
    reward_component_weights: dict[str, float] | None = None,
    move_tensors_to_cpu: bool = True,
    reward_fn: Callable[[dict[str, Any]], Any] | None = None,
    parallel_rollout_workers: int = 1,
    rollout_envs: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if device is None:
        raise RuntimeError("`device` must be provided for collect_group_from_anchor.")
    if group_size <= 0:
        raise RuntimeError("`group_size` must be > 0.")

    obs = env.reset_to_snapshot(anchor_snapshot)
    obs_tensor = _to_obs_tensor(obs, device=device)

    prepared_predictions: list[dict[str, Any]] = []
    for _ in range(group_size):
        prepared_predictions.append(
            _prepare_branch_prediction_from_obs(
                model=model,
                obs_tensor=obs_tensor,
                action_normalizer=action_normalizer,
                goal_obs_dict=goal_obs_dict,
                decode_pred_obs=decode_pred_obs,
                require_logprob=require_logprob,
                move_tensors_to_cpu=move_tensors_to_cpu,
            )
        )

    rollout_outputs: list[dict[str, Any]] = [None] * group_size  # type: ignore[list-item]
    num_workers = max(1, int(parallel_rollout_workers))
    rollout_env_list = list(rollout_envs) if rollout_envs is not None else None
    use_parallel_rollout = num_workers > 1

    if use_parallel_rollout:
        if rollout_env_list is None or len(rollout_env_list) == 0:
            raise RuntimeError(
                "parallel_rollout_workers > 1 but `rollout_envs` is empty. "
                "Please create parallel env instances and pass `rollout_envs`."
            )
        num_workers = min(num_workers, len(rollout_env_list), group_size)
        for start in range(0, group_size, num_workers):
            end = min(start + num_workers, group_size)
            with ThreadPoolExecutor(max_workers=(end - start)) as executor:
                futures = []
                for slot, branch_idx in enumerate(range(start, end)):
                    futures.append(
                        executor.submit(
                            _rollout_from_anchor,
                            rollout_env=rollout_env_list[slot],
                            anchor_snapshot=anchor_snapshot,
                            action_exec=prepared_predictions[branch_idx]["action_exec"],
                            terminate_on_success=terminate_on_success,
                        )
                    )
                for slot, fut in enumerate(futures):
                    rollout_outputs[start + slot] = fut.result()
    else:
        for branch_idx in range(group_size):
            rollout_outputs[branch_idx] = _rollout_from_anchor(
                rollout_env=env,
                anchor_snapshot=anchor_snapshot,
                action_exec=prepared_predictions[branch_idx]["action_exec"],
                terminate_on_success=terminate_on_success,
            )

    branches = []
    for branch_id in range(group_size):
        branch = _build_branch_record_from_prediction_and_rollout(
            anchor_snapshot=anchor_snapshot,
            obs_cond=obs,
            prediction=prepared_predictions[branch_id],
            rollout=rollout_outputs[branch_id],
            reward_fn=reward_fn,
            group_id=group_id,
            branch_id=branch_id,
        )
        branches.append(branch)

    is_complete, drop_reason = validate_group_complete(
        branches, required_action_horizon=required_action_horizon
    )
    is_valid = (not drop_if_any_incomplete) or is_complete
    group_record = {
        "group_id": group_id,
        "anchor_snapshot": copy.deepcopy(anchor_snapshot),
        "branches": branches,
        "required_action_horizon": required_action_horizon,
        "is_complete": is_complete,
        "is_valid": is_valid,
        "drop_reason": None if is_valid else drop_reason,
    }
    if is_valid and compute_group_advantage:
        build_group_advantages(
            group_record,
            strict_dance_std=strict_dance_std,
            rank_transform=rank_transform,
            rank_transform_components=rank_transform_components,
            component_norm=reward_component_norm,
            component_keys=reward_component_keys,
            component_weights=reward_component_weights,
        )
    return group_record


def update_state_pool_from_group(state_pool: StatePool, group_record: dict[str, Any]):
    if not group_record["is_valid"]:
        return
    final_snapshots = []
    for branch in group_record["branches"]:
        final_snapshot = branch.get("final_snapshot", None)
        if final_snapshot is None:
            continue
        # Require simulatable state for future reset.
        if final_snapshot.get("sim_state", None) is None:
            continue
        final_snapshots.append(final_snapshot)
    state_pool.extend(final_snapshots)


def build_group_advantages(
    group_record: dict[str, Any],
    eps: float = 1e-8,
    strict_dance_std: bool = True,
    component_norm: bool = False,
    component_keys: Sequence[str] | None = None,
    component_weights: dict[str, float] | None = None,
    component_metric_prefix: str = "metrics/",
    rank_transform: str = "none",
    rank_transform_components: bool = True,
) -> dict[str, Any]:
    """
    Normalize rewards within a group and store branch-level GRPO advantages.
    DanceGRPO-consistent path:
      adv = (reward - mean_group) / (std_group + 1e-8), with std(unbiased=True).
    """
    branches = group_record.get("branches", [])
    if len(branches) == 0:
        group_record["reward_mean"] = 0.0
        group_record["reward_std"] = 0.0
        return group_record

    rewards = torch.tensor(
        [float(branch.get("reward", branch.get("reward_real", 0.0))) for branch in branches],
        dtype=torch.float32,
    )
    transformed_rewards = _rank_transform_values(rewards, mode=rank_transform, eps=eps)
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=True if strict_dance_std else False)
    transformed_std = transformed_rewards.std(unbiased=True if strict_dance_std else False)
    if (not torch.isfinite(transformed_std)) or float(transformed_std.item()) < eps:
        scalar_advantages = torch.zeros_like(transformed_rewards)
    else:
        transformed_mean = transformed_rewards.mean()
        scalar_advantages = (transformed_rewards - transformed_mean) / (transformed_std + eps)

    fused_advantages = scalar_advantages
    component_advantages: dict[str, torch.Tensor] = {}
    component_rewards: dict[str, torch.Tensor] = {}
    component_alpha: dict[str, float] = {}
    used_component_norm = False

    if component_norm:
        keys = list(component_keys) if component_keys is not None else None
        if keys is None:
            if isinstance(component_weights, dict):
                keys = [k for k, w in component_weights.items() if abs(float(w)) > 0.0]
            else:
                # Infer from reward details if weights are not provided.
                inferred = set()
                for branch in branches:
                    details = branch.get("reward_details", {})
                    if not isinstance(details, dict):
                        continue
                    for k in details.keys():
                        if isinstance(k, str) and k.startswith(component_metric_prefix):
                            inferred.add(k[len(component_metric_prefix) :])
                keys = sorted(inferred)

        if keys:
            for key in keys:
                metric_key = f"{component_metric_prefix}{key}"
                vals = []
                ok = True
                for branch in branches:
                    details = branch.get("reward_details", {})
                    if not isinstance(details, dict) or metric_key not in details:
                        ok = False
                        break
                    v = details[metric_key]
                    if not isinstance(v, (int, float)) or (not np.isfinite(float(v))):
                        ok = False
                        break
                    # Reward component follows reward=-loss convention.
                    vals.append(-float(v))
                if not ok:
                    continue

                comp_reward = torch.tensor(vals, dtype=torch.float32)
                comp_reward_transformed = (
                    _rank_transform_values(comp_reward, mode=rank_transform, eps=eps)
                    if rank_transform_components
                    else comp_reward
                )
                comp_mean = comp_reward_transformed.mean()
                comp_std = comp_reward_transformed.std(unbiased=True if strict_dance_std else False)
                if (not torch.isfinite(comp_std)) or float(comp_std.item()) < eps:
                    comp_adv = torch.zeros_like(comp_reward_transformed)
                else:
                    comp_adv = (comp_reward_transformed - comp_mean) / (comp_std + eps)

                component_rewards[key] = comp_reward
                component_advantages[key] = comp_adv

            if len(component_advantages) > 0:
                key_list = list(component_advantages.keys())
                if isinstance(component_weights, dict):
                    alpha_t = torch.tensor(
                        [abs(float(component_weights.get(k, 0.0))) for k in key_list],
                        dtype=torch.float32,
                    )
                else:
                    alpha_t = torch.ones(len(key_list), dtype=torch.float32)
                if float(alpha_t.sum().item()) <= eps:
                    alpha_t = torch.ones(len(key_list), dtype=torch.float32)
                alpha_t = alpha_t / alpha_t.sum()

                fused_advantages = torch.zeros_like(rewards)
                for idx, key in enumerate(key_list):
                    a = float(alpha_t[idx].item())
                    fused_advantages = fused_advantages + a * component_advantages[key]
                    component_alpha[key] = a

                used_component_norm = True

    for idx, branch in enumerate(branches):
        adv_value = float(fused_advantages[idx].item())
        branch["advantage"] = adv_value
        branch["final_advantage"] = adv_value
        branch["advantage_scalar_fallback"] = float(scalar_advantages[idx].item())
        if used_component_norm:
            branch["advantage_components"] = {
                k: float(component_advantages[k][idx].item()) for k in component_advantages.keys()
            }
            branch["reward_components"] = {
                k: float(component_rewards[k][idx].item()) for k in component_rewards.keys()
            }
        else:
            branch["advantage_components"] = {}
            branch["reward_components"] = {}

        ret_value = float(branch.get("reward", branch.get("reward_real", 0.0)))
        branch["returns"] = ret_value
        old_lp = branch.get("old_log_probs", None)
        if old_lp is not None:
            old_lp_t = _to_tensor(old_lp, dtype=torch.float32)
            branch["response_mask"] = torch.ones_like(old_lp_t, dtype=torch.float32)
            branch["advantages_t"] = torch.full_like(old_lp_t, adv_value, dtype=torch.float32)
            branch["returns_t"] = torch.full_like(old_lp_t, ret_value, dtype=torch.float32)

    group_record["reward_mean"] = float(reward_mean.item())
    group_record["reward_std"] = float(reward_std.item())
    group_record["reward_mean_transformed"] = float(transformed_rewards.mean().item())
    group_record["reward_std_transformed"] = float(transformed_std.item())
    group_record["rank_transform"] = str(rank_transform)
    group_record["component_norm_used"] = bool(used_component_norm)
    group_record["component_alpha"] = component_alpha
    return group_record


@torch.no_grad()
def collect_online_groups(
    env,
    model,
    initial_snapshot: dict[str, Any],
    num_groups: int,
    group_size: int = DEFAULT_GRPO_GROUP_SIZE,
    device: torch.device | None = None,
    action_normalizer=None,
    goal_obs_dict: dict[str, torch.Tensor] | None = None,
    required_action_horizon: int = 16,
    drop_if_any_incomplete: bool = True,
    terminate_on_success: bool = False,
    decode_pred_obs: bool = True,
    require_logprob: bool = True,
    compute_group_advantage: bool = True,
    strict_dance_std: bool = True,
    rank_transform: str = "none",
    rank_transform_components: bool = True,
    reward_fn: Callable[[dict[str, Any]], Any] | None = None,
    rlvr_reward_model=None,
    reward_obs_key: str | None = None,
    reward_obs_keys: Sequence[str] | None = None,
    reward_pred_view_index: int = 0,
    reward_use_multi_view: bool = False,
    reward_strict_num_views: bool = False,
    reward_use_full_time: bool = True,
    reward_time_align: str = "tail",
    reward_pred_time_index: int = -1,
    reward_real_time_index: int = -1,
    reward_aggregate: str = "mean",
    reward_time_aggregate: str = "mean",
    reward_view_aggregate: str = "mean",
    reward_layout_5d: str = "btchw",
    reward_layout_6d: str = "auto",
    reward_clamp_pred: bool = True,
    reward_component_norm: bool = False,
    reward_component_keys: Sequence[str] | None = None,
    reward_component_weights: dict[str, float] | None = None,
    move_tensors_to_cpu: bool = True,
    state_pool_max_size: int = 4096,
    seed: int = 0,
    parallel_rollout_workers: int = 1,
    rollout_envs: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if device is None:
        raise RuntimeError("`device` must be provided for collect_online_groups.")
    state_pool = StatePool(max_size=state_pool_max_size, seed=seed)
    state_pool.push(initial_snapshot)
    groups = []

    effective_reward_fn = reward_fn
    if effective_reward_fn is None and rlvr_reward_model is not None:
        # For multi-view reward, prefer runtime keys from next_obs_real inside
        # reward_fn to avoid key-space mismatch between snapshot/raw obs and
        # wrapped env outputs (e.g. agentview_image vs agentview_rgb).
        obs_keys = reward_obs_keys
        obs_key = reward_obs_key
        if not reward_use_multi_view:
            obs_key = reward_obs_key or _infer_reward_obs_key(initial_snapshot)
            if obs_key is None:
                raise RuntimeError(
                    "Cannot infer `reward_obs_key` from initial_snapshot. "
                    "Please pass `reward_obs_key` explicitly."
                )
        effective_reward_fn = make_rlvr_image_reward_fn(
            rlvr_reward_model=rlvr_reward_model,
            reward_obs_key=obs_key,
            reward_obs_keys=obs_keys,
            pred_view_index=reward_pred_view_index,
            use_multi_view=reward_use_multi_view,
            strict_num_views=reward_strict_num_views,
            use_full_time=reward_use_full_time,
            time_align=reward_time_align,
            pred_time_index=reward_pred_time_index,
            real_time_index=reward_real_time_index,
            aggregate=reward_aggregate,
            time_aggregate=reward_time_aggregate,
            view_aggregate=reward_view_aggregate,
            layout_5d=reward_layout_5d,
            layout_6d=reward_layout_6d,
            clamp_pred=reward_clamp_pred,
        )
    effective_decode_pred_obs = decode_pred_obs
    if rlvr_reward_model is not None and not effective_decode_pred_obs:
        effective_decode_pred_obs = True

    for group_id in range(num_groups):
        anchor_snapshot = state_pool.sample_one()
        group_record = collect_group_from_anchor(
            env=env,
            model=model,
            anchor_snapshot=anchor_snapshot,
            group_size=group_size,
            device=device,
            action_normalizer=action_normalizer,
            goal_obs_dict=goal_obs_dict,
            required_action_horizon=required_action_horizon,
            drop_if_any_incomplete=drop_if_any_incomplete,
            terminate_on_success=terminate_on_success,
            decode_pred_obs=effective_decode_pred_obs,
            require_logprob=require_logprob,
            group_id=group_id,
            compute_group_advantage=compute_group_advantage,
            strict_dance_std=strict_dance_std,
            rank_transform=rank_transform,
            rank_transform_components=rank_transform_components,
            reward_component_norm=reward_component_norm,
            reward_component_keys=reward_component_keys,
            reward_component_weights=reward_component_weights,
            move_tensors_to_cpu=move_tensors_to_cpu,
            reward_fn=effective_reward_fn,
            parallel_rollout_workers=parallel_rollout_workers,
            rollout_envs=rollout_envs,
        )
        groups.append(group_record)
        update_state_pool_from_group(state_pool, group_record)

    num_valid_groups = sum(1 for group in groups if group.get("is_valid", False))
    if reward_fn is not None:
        reward_mode = "custom_reward_fn"
    elif rlvr_reward_model is not None:
        reward_mode = "rlvr_image"
    else:
        reward_mode = "env"
    return {
        "groups": groups,
        "num_groups_collected": len(groups),
        "num_valid_groups": num_valid_groups,
        "state_pool_size": len(state_pool),
        "reward_mode": reward_mode,
    }


def flatten_groups_to_grpo_batch(
    groups: list[dict[str, Any]],
    drop_invalid_groups: bool = True,
    require_advantage: bool = True,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Flatten a list of group records into GRPO-compatible batch tensors.
    Required branch keys:
      timesteps, latents, next_latents, old_log_probs, reward
    Optional:
      advantage, action_pred, anchor_snapshot, obs_cond
    """
    selected_groups = []
    for group in groups:
        if drop_invalid_groups and (not group.get("is_valid", False)):
            continue
        selected_groups.append(group)

    flat_branches = []
    for group in selected_groups:
        for branch in group.get("branches", []):
            flat_branches.append(branch)

    if len(flat_branches) == 0:
        raise RuntimeError("No branches available to build GRPO batch.")

    def _stack_required_tensor(key: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        values = []
        for idx, branch in enumerate(flat_branches):
            value = branch.get(key, None)
            if value is None:
                raise RuntimeError(f"Missing required key `{key}` in flattened branch #{idx}.")
            values.append(_to_tensor(value, dtype=dtype))
        return torch.stack(values, dim=0)

    timesteps = _stack_required_tensor("timesteps", dtype=torch.long)
    latents = _stack_required_tensor("latents")
    next_latents = _stack_required_tensor("next_latents")
    old_log_probs = _stack_required_tensor("old_log_probs")
    log_probs = old_log_probs

    rewards = torch.tensor(
        [float(branch.get("reward", branch.get("reward_real", 0.0))) for branch in flat_branches],
        dtype=torch.float32,
    )

    if require_advantage:
        missing_adv = [i for i, branch in enumerate(flat_branches) if "advantage" not in branch]
        if len(missing_adv) > 0:
            raise RuntimeError(
                f"Advantages are required but missing in {len(missing_adv)} flattened branches."
            )
        advantages = torch.tensor(
            [float(branch["advantage"]) for branch in flat_branches],
            dtype=torch.float32,
        )
    else:
        advantages = torch.tensor(
            [float(branch.get("advantage", 0.0)) for branch in flat_branches],
            dtype=torch.float32,
        )
    returns = torch.tensor(
        [float(branch.get("returns", branch.get("reward", branch.get("reward_real", 0.0)))) for branch in flat_branches],
        dtype=torch.float32,
    )

    advantages_t_list = []
    returns_t_list = []
    response_mask_list = []
    for idx, branch in enumerate(flat_branches):
        old_lp = _to_tensor(branch["old_log_probs"], dtype=torch.float32)
        response_mask_t = _to_tensor(
            branch.get("response_mask", torch.ones_like(old_lp, dtype=torch.float32)),
            dtype=torch.float32,
        )
        if response_mask_t.shape != old_lp.shape:
            raise RuntimeError(
                f"response_mask shape mismatch at branch #{idx}: {tuple(response_mask_t.shape)} vs {tuple(old_lp.shape)}"
            )

        adv_t = branch.get("advantages_t", None)
        if adv_t is None:
            adv_t = torch.full_like(old_lp, float(branch.get("advantage", 0.0)), dtype=torch.float32)
        else:
            adv_t = _to_tensor(adv_t, dtype=torch.float32)
        if adv_t.shape != old_lp.shape:
            raise RuntimeError(
                f"advantages_t shape mismatch at branch #{idx}: {tuple(adv_t.shape)} vs {tuple(old_lp.shape)}"
            )

        ret_t = branch.get("returns_t", None)
        if ret_t is None:
            ret_t = torch.full_like(
                old_lp,
                float(branch.get("returns", branch.get("reward", branch.get("reward_real", 0.0)))),
                dtype=torch.float32,
            )
        else:
            ret_t = _to_tensor(ret_t, dtype=torch.float32)
        if ret_t.shape != old_lp.shape:
            raise RuntimeError(
                f"returns_t shape mismatch at branch #{idx}: {tuple(ret_t.shape)} vs {tuple(old_lp.shape)}"
            )

        response_mask_list.append(response_mask_t)
        advantages_t_list.append(adv_t)
        returns_t_list.append(ret_t)

    response_mask = torch.stack(response_mask_list, dim=0)
    advantages_t = torch.stack(advantages_t_list, dim=0)
    returns_t = torch.stack(returns_t_list, dim=0)

    group_ids = torch.tensor(
        [
            -1 if branch.get("group_id", None) is None else int(branch["group_id"])
            for branch in flat_branches
        ],
        dtype=torch.long,
    )
    branch_ids = torch.tensor(
        [
            -1 if branch.get("branch_id", None) is None else int(branch["branch_id"])
            for branch in flat_branches
        ],
        dtype=torch.long,
    )

    action_pred = None
    if all(branch.get("action_pred", None) is not None for branch in flat_branches):
        action_pred = torch.stack([_to_tensor(branch["action_pred"]) for branch in flat_branches], dim=0)

    out = {
        "timesteps": timesteps,
        "latents": latents,
        "next_latents": next_latents,
        "old_log_probs": old_log_probs,
        "log_probs": log_probs,
        "rewards": rewards,
        "returns": returns,
        "advantages": advantages,
        "advantages_t": advantages_t,
        "returns_t": returns_t,
        "response_mask": response_mask,
        "group_ids": group_ids,
        "branch_ids": branch_ids,
        "action_pred": action_pred,
        "anchor_snapshot": [copy.deepcopy(branch.get("anchor_snapshot", None)) for branch in flat_branches],
        "obs_cond": [copy.deepcopy(branch.get("obs_cond", None)) for branch in flat_branches],
    }
    if device is not None:
        target_device = torch.device(device)
        for key in (
            "timesteps",
            "latents",
            "next_latents",
            "old_log_probs",
            "log_probs",
            "rewards",
            "returns",
            "advantages",
            "advantages_t",
            "returns_t",
            "response_mask",
            "group_ids",
            "branch_ids",
        ):
            out[key] = out[key].to(target_device, non_blocking=True)
        if out["action_pred"] is not None:
            out["action_pred"] = out["action_pred"].to(target_device, non_blocking=True)
    return out


@torch.no_grad()
def collect_online_groups_as_grpo_batch(
    env,
    model,
    initial_snapshot: dict[str, Any],
    num_groups: int,
    group_size: int = DEFAULT_GRPO_GROUP_SIZE,
    device: torch.device | None = None,
    action_normalizer=None,
    goal_obs_dict: dict[str, torch.Tensor] | None = None,
    required_action_horizon: int = 16,
    drop_if_any_incomplete: bool = True,
    terminate_on_success: bool = False,
    decode_pred_obs: bool = True,
    require_logprob: bool = True,
    compute_group_advantage: bool = True,
    strict_dance_std: bool = True,
    rank_transform: str = "none",
    rank_transform_components: bool = True,
    reward_fn: Callable[[dict[str, Any]], Any] | None = None,
    rlvr_reward_model=None,
    reward_obs_key: str | None = None,
    reward_obs_keys: Sequence[str] | None = None,
    reward_pred_view_index: int = 0,
    reward_use_multi_view: bool = False,
    reward_strict_num_views: bool = False,
    reward_use_full_time: bool = True,
    reward_time_align: str = "tail",
    reward_pred_time_index: int = -1,
    reward_real_time_index: int = -1,
    reward_aggregate: str = "mean",
    reward_time_aggregate: str = "mean",
    reward_view_aggregate: str = "mean",
    reward_layout_5d: str = "btchw",
    reward_layout_6d: str = "auto",
    reward_clamp_pred: bool = True,
    reward_component_norm: bool = False,
    reward_component_keys: Sequence[str] | None = None,
    reward_component_weights: dict[str, float] | None = None,
    move_tensors_to_cpu: bool = True,
    state_pool_max_size: int = 4096,
    seed: int = 0,
    batch_device: torch.device | str | None = None,
    parallel_rollout_workers: int = 1,
    rollout_envs: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if device is None:
        raise RuntimeError("`device` must be provided for collect_online_groups_as_grpo_batch.")
    result = collect_online_groups(
        env=env,
        model=model,
        initial_snapshot=initial_snapshot,
        num_groups=num_groups,
        group_size=group_size,
        device=device,
        action_normalizer=action_normalizer,
        goal_obs_dict=goal_obs_dict,
        required_action_horizon=required_action_horizon,
        drop_if_any_incomplete=drop_if_any_incomplete,
        terminate_on_success=terminate_on_success,
        decode_pred_obs=decode_pred_obs,
        require_logprob=require_logprob,
        compute_group_advantage=compute_group_advantage,
        strict_dance_std=strict_dance_std,
        rank_transform=rank_transform,
        rank_transform_components=rank_transform_components,
        reward_fn=reward_fn,
        rlvr_reward_model=rlvr_reward_model,
        reward_obs_key=reward_obs_key,
        reward_obs_keys=reward_obs_keys,
        reward_pred_view_index=reward_pred_view_index,
        reward_use_multi_view=reward_use_multi_view,
        reward_strict_num_views=reward_strict_num_views,
        reward_use_full_time=reward_use_full_time,
        reward_time_align=reward_time_align,
        reward_pred_time_index=reward_pred_time_index,
        reward_real_time_index=reward_real_time_index,
        reward_aggregate=reward_aggregate,
        reward_time_aggregate=reward_time_aggregate,
        reward_view_aggregate=reward_view_aggregate,
        reward_layout_5d=reward_layout_5d,
        reward_layout_6d=reward_layout_6d,
        reward_clamp_pred=reward_clamp_pred,
        reward_component_norm=reward_component_norm,
        reward_component_keys=reward_component_keys,
        reward_component_weights=reward_component_weights,
        move_tensors_to_cpu=move_tensors_to_cpu,
        state_pool_max_size=state_pool_max_size,
        seed=seed,
        parallel_rollout_workers=parallel_rollout_workers,
        rollout_envs=rollout_envs,
    )
    result["grpo_batch"] = flatten_groups_to_grpo_batch(
        result["groups"],
        drop_invalid_groups=drop_if_any_incomplete,
        require_advantage=compute_group_advantage,
        device=batch_device,
    )
    return result

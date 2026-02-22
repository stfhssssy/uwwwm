from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


LOSS_KEYS = ("lpips", "mse", "mae", "ssim", "psnr")
DEFAULT_LOSS_WEIGHT = {
    "lpips": 1.0,
    "mse": 0.0,
    "mae": 1.0,
    "ssim": 0.0,
    "psnr": 0.0,
}


def _normalize_loss_weight(loss_weight: Dict[str, float] | None) -> Dict[str, float]:
    if loss_weight is None:
        loss_weight = DEFAULT_LOSS_WEIGHT
    merged = dict(DEFAULT_LOSS_WEIGHT)
    merged.update(loss_weight)
    return {k: float(merged[k]) for k in LOSS_KEYS}


def _load_rlvr_lpips_class(lpips_py_path: str | None = None):
    if lpips_py_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        lpips_py_path = str(repo_root / "RLVR-World" / "vid_wm" / "verl" / "ivideogpt" / "lpips.py")
    lpips_file = Path(lpips_py_path)
    if not lpips_file.exists():
        raise FileNotFoundError(f"RLVR LPIPS file not found: {lpips_file}")

    module_name = f"rlvr_lpips_{lpips_file.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, str(lpips_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec from {lpips_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "LPIPS"):
        raise AttributeError(f"`LPIPS` class not found in {lpips_file}")
    return module.LPIPS


def _aggregate_over_time(x_bt: torch.Tensor, mode: str = "mean", discount: float = 0.9) -> torch.Tensor:
    if mode == "mean":
        return x_bt.mean(dim=1)
    if mode == "last":
        return x_bt[:, -1]
    if mode == "discount":
        weight = discount ** torch.arange(x_bt.shape[1], device=x_bt.device, dtype=x_bt.dtype)
        return (x_bt * weight.unsqueeze(0)).sum(dim=1) / weight.sum()
    raise ValueError(f"Unsupported aggregate mode: {mode}")


def _aggregate_over_axis(
    x: torch.Tensor,
    axis: int,
    mode: str = "mean",
    discount: float = 0.9,
) -> torch.Tensor:
    if mode == "mean":
        return x.mean(dim=axis)
    if mode == "last":
        return x.select(dim=axis, index=x.shape[axis] - 1)
    if mode == "discount":
        size = x.shape[axis]
        weight = discount ** torch.arange(size, device=x.device, dtype=x.dtype)
        shape = [1] * x.ndim
        shape[axis] = size
        weighted = x * weight.reshape(shape)
        return weighted.sum(dim=axis) / weight.sum()
    raise ValueError(f"Unsupported aggregate mode: {mode}")


class RLVRImageReward(nn.Module):
    """
    RLVR-compatible image reward:
      reward = - loss
      loss = sum_k( loss_weight[k] * metric_k )

    Metric formulas are aligned with:
      - `TokenizerWorker._perceptual_loss`
      - `TokenizerWorker._compute_loss`
    in RLVR-World.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        loss_weight: Dict[str, float] | None = None,
        lpips_micro_batch_size: int = 8,
        lpips_py_path: str | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.loss_weight = _normalize_loss_weight(loss_weight)
        self.lpips_micro_batch_size = int(lpips_micro_batch_size)

        self.lpips = None
        if self.loss_weight["lpips"]:
            LPIPS = _load_rlvr_lpips_class(lpips_py_path=lpips_py_path)
            self.lpips = LPIPS().to(self.device).eval()
        self.psnr = None
        self.ssim = None
        if self.loss_weight["ssim"] or self.loss_weight["psnr"]:
            try:
                import piqa
            except ImportError as e:
                raise RuntimeError(
                    "`piqa` is required when loss_weight['ssim'] or loss_weight['psnr'] is non-zero."
                ) from e
            self.psnr = piqa.PSNR(epsilon=1e-08, value_range=1.0, reduction="none").to(self.device)
            self.ssim = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction="none").to(self.device)

    @torch.no_grad()
    def _perceptual_loss(self, real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.lpips is None:
            raise RuntimeError("LPIPS is not initialized. Set loss_weight['lpips'] > 0.")
        bs = self.lpips_micro_batch_size
        losses = []
        use_autocast = self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            for i in range(0, real.shape[0], bs):
                l = self.lpips(
                    real[i : i + bs].contiguous() * 2 - 1.0,
                    pred[i : i + bs].contiguous() * 2 - 1.0,
                ).mean(dim=(1, 2, 3))
                losses.append(l)
        return torch.cat(losses, dim=0)

    @torch.no_grad()
    def _compute_loss(self, real: torch.Tensor, pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        # real/pred: [N, C, H, W], expected in [0, 1]
        out: Dict[str, torch.Tensor] = {}
        loss = torch.zeros(real.shape[0], device=real.device, dtype=real.dtype)

        if self.loss_weight["lpips"]:
            lpips = self._perceptual_loss(real, pred)
            out["lpips"] = lpips
            loss = loss + self.loss_weight["lpips"] * lpips

        if self.loss_weight["ssim"]:
            if self.ssim is None:
                raise RuntimeError("SSIM metric is not initialized. Install `piqa` and enable loss_weight['ssim'].")
            ssim = -self.ssim(real.float(), pred.float())
            out["ssim"] = ssim
            loss = loss + self.loss_weight["ssim"] * ssim

        if self.loss_weight["psnr"]:
            if self.psnr is None:
                raise RuntimeError("PSNR metric is not initialized. Install `piqa` and enable loss_weight['psnr'].")
            psnr = -self.psnr(real.float(), pred.float())
            out["psnr"] = psnr
            loss = loss + self.loss_weight["psnr"] * psnr

        if self.loss_weight["mse"]:
            mse = torch.mean((real - pred) ** 2, dim=(1, 2, 3))
            out["mse"] = mse
            loss = loss + self.loss_weight["mse"] * mse

        if self.loss_weight["mae"]:
            mae = torch.mean(torch.abs(real - pred), dim=(1, 2, 3))
            out["mae"] = mae
            loss = loss + self.loss_weight["mae"] * mae

        out["loss"] = loss
        return out

    @torch.no_grad()
    def forward(
        self,
        real: torch.Tensor,
        pred: torch.Tensor,
        aggregate: str | None = None,
        discount: float = 0.9,
        time_aggregate: str = "mean",
        view_aggregate: str = "mean",
        view_discount: float = 0.9,
        layout_5d: str = "btchw",
        layout_6d: str = "auto",
        clamp_pred: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            real, pred:
              - [B, C, H, W] (single-step)
              - [B, T, C, H, W] (multi-step, `layout_5d=btchw`)
              - [B, V, C, H, W] (multi-view, `layout_5d=bvchw`)
              - [B, V, T, C, H, W] / [B, V, C, T, H, W] (multi-view multi-step)
            aggregate:
              Backward-compatible alias for `time_aggregate`.
        """
        if real.shape != pred.shape:
            raise ValueError(f"`real` and `pred` must have same shape, got {tuple(real.shape)} vs {tuple(pred.shape)}")
        if real.ndim not in (4, 5, 6):
            raise ValueError(f"Expected 4D/5D/6D image tensor, got shape={tuple(real.shape)}")

        real = real.to(self.device, non_blocking=True)
        pred = pred.to(self.device, non_blocking=True)
        if clamp_pred:
            pred = pred.clamp(0.0, 1.0)

        if aggregate is not None:
            time_aggregate = aggregate

        if real.ndim == 4:
            metrics = self._compute_loss(real, pred)
            return {
                "reward": -metrics["loss"],
                "loss": metrics["loss"],
                "metrics": metrics,
                "frame_metrics": None,
                "view_metrics": None,
            }

        # 5D mode: either [B, T, C, H, W] or [B, V, C, H, W]
        if real.ndim == 5:
            if layout_5d == "btchw":
                b, t = real.shape[:2]
                metrics_bt_flat = self._compute_loss(
                    real.reshape(b * t, *real.shape[-3:]),
                    pred.reshape(b * t, *pred.shape[-3:]),
                )
                frame_metrics = {k: v.reshape(b, t) for k, v in metrics_bt_flat.items()}
                metrics = {
                    k: _aggregate_over_time(v_bt, mode=time_aggregate, discount=discount)
                    for k, v_bt in frame_metrics.items()
                }
                return {
                    "reward": -metrics["loss"],
                    "loss": metrics["loss"],
                    "metrics": metrics,
                    "frame_metrics": frame_metrics,
                    "view_metrics": None,
                }
            if layout_5d == "bvchw":
                b, v = real.shape[:2]
                metrics_bv_flat = self._compute_loss(
                    real.reshape(b * v, *real.shape[-3:]),
                    pred.reshape(b * v, *pred.shape[-3:]),
                )
                view_metrics = {k: m.reshape(b, v) for k, m in metrics_bv_flat.items()}
                metrics = {
                    k: _aggregate_over_axis(v_bv, axis=1, mode=view_aggregate, discount=view_discount)
                    for k, v_bv in view_metrics.items()
                }
                return {
                    "reward": -metrics["loss"],
                    "loss": metrics["loss"],
                    "metrics": metrics,
                    "frame_metrics": None,
                    "view_metrics": view_metrics,
                }
            raise ValueError(f"Unsupported layout_5d={layout_5d}; choose from ['btchw', 'bvchw']")

        # 6D mode: [B, V, T, C, H, W] or [B, V, C, T, H, W]
        if layout_6d == "auto":
            if real.shape[3] == 3:
                real_bvtchw = real
                pred_bvtchw = pred
            elif real.shape[2] == 3:
                real_bvtchw = real.permute(0, 1, 3, 2, 4, 5).contiguous()
                pred_bvtchw = pred.permute(0, 1, 3, 2, 4, 5).contiguous()
            else:
                raise ValueError(
                    "Unable to infer 6D layout automatically. "
                    "Set layout_6d to 'bvtchw' or 'bvcthw'."
                )
        elif layout_6d == "bvtchw":
            real_bvtchw = real
            pred_bvtchw = pred
        elif layout_6d == "bvcthw":
            real_bvtchw = real.permute(0, 1, 3, 2, 4, 5).contiguous()
            pred_bvtchw = pred.permute(0, 1, 3, 2, 4, 5).contiguous()
        else:
            raise ValueError(f"Unsupported layout_6d={layout_6d}; choose from ['auto', 'bvtchw', 'bvcthw']")

        b, v, t = real_bvtchw.shape[:3]
        metrics_bvt_flat = self._compute_loss(
            real_bvtchw.reshape(b * v * t, *real_bvtchw.shape[-3:]),
            pred_bvtchw.reshape(b * v * t, *pred_bvtchw.shape[-3:]),
        )
        frame_metrics = {k: m.reshape(b, v, t) for k, m in metrics_bvt_flat.items()}
        view_metrics = {
            k: _aggregate_over_axis(v_bvt, axis=2, mode=time_aggregate, discount=discount)
            for k, v_bvt in frame_metrics.items()
        }
        metrics = {
            k: _aggregate_over_axis(v_bv, axis=1, mode=view_aggregate, discount=view_discount)
            for k, v_bv in view_metrics.items()
        }
        return {
            "reward": -metrics["loss"],
            "loss": metrics["loss"],
            "metrics": metrics,
            "frame_metrics": frame_metrics,
            "view_metrics": view_metrics,
        }

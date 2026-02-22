#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise RuntimeError(
            "This script requires PyTorch. Please run it in an environment with `torch` installed."
        ) from e
    return torch, F


DISPLAY_KEY = {
    "taco_play": "rgb_static",
    "roboturk": "front_rgb",
    "viola": "agentview_rgb",
    "berkeley_autolab_ur5": "hand_image",
    "language_table": "rgb",
    "berkeley_mvp_converted_externally_to_rlds": "hand_image",
    "berkeley_rpt_converted_externally_to_rlds": "hand_image",
    "stanford_robocook_converted_externally_to_rlds1": "image_1",
    "stanford_robocook_converted_externally_to_rlds2": "image_2",
    "stanford_robocook_converted_externally_to_rlds3": "image_3",
    "stanford_robocook_converted_externally_to_rlds4": "image_4",
    "uiuc_d3field1": "image_1",
    "uiuc_d3field2": "image_2",
    "uiuc_d3field3": "image_3",
    "uiuc_d3field4": "image_4",
}


def get_display_key(dataset_name: str) -> str:
    return DISPLAY_KEY.get(dataset_name, "image")


def extract_goal_image(frames: Any) -> Any:
    # frames: [B, T, C, H, W] or [T, C, H, W]
    if frames.ndim == 5:
        return frames[:, -1]
    if frames.ndim == 4:
        return frames[-1]
    raise ValueError(f"Unsupported shape for `frames`: {tuple(frames.shape)}")


def extract_initial_image(frames: Any) -> Any:
    # frames: [B, T, C, H, W] or [T, C, H, W]
    if frames.ndim == 5:
        return frames[:, 0]
    if frames.ndim == 4:
        return frames[0]
    raise ValueError(f"Unsupported shape for `frames`: {tuple(frames.shape)}")


@dataclass
class SegmentMeta:
    file_path: str
    episode_length: int
    start: int
    stepsize: int
    frame_indices: List[int]


class RLVRLikeLoader:
    def __init__(
        self,
        dataset_path: str,
        dataset_name: str,
        split: str,
        segment_length: int,
        context_length: int,
        stepsize: int,
        random_selection: bool,
        segment_horizon: Optional[int],
        from_start: bool,
        random_ctx_frame: bool,
        image_size: int,
        load_action: bool,
        seed: int,
    ):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.segment_length = segment_length
        self.context_length = context_length
        self.stepsize = max(1, stepsize)
        self.random_selection = random_selection
        self.segment_horizon = segment_horizon or segment_length
        self.from_start = from_start
        self.random_ctx_frame = random_ctx_frame
        self.image_size = image_size
        self.load_action = load_action

        self.display_key = get_display_key(dataset_name)
        self.rng = np.random.default_rng(seed)
        self.files = self._discover_files()

        if not self.files:
            raise FileNotFoundError(
                f"No usable episodes under {dataset_path}/{dataset_name} for split={split}."
            )

    def _discover_files(self) -> List[str]:
        files = sorted(glob.glob(os.path.join(self.dataset_path, self.dataset_name, "*.npz")))
        if self.split == "train":
            files = [x for i, x in enumerate(files) if i % 100 != 0]
        elif self.split == "val":
            files = [x for i, x in enumerate(files) if i % 100 == 0]
        else:
            raise ValueError(f"Unsupported split={self.split}. Use `train` or `val`.")
        return files

    def _load_episode(self, file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        with np.load(file_path) as episode:
            if self.display_key not in episode.files:
                raise KeyError(
                    f"`{self.display_key}` not in {file_path}. Available keys: {list(episode.files)}"
                )
            frames = episode[self.display_key]
            actions = episode["action"] if ("action" in episode.files and self.load_action) else None

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected [T,H,W,3], got {frames.shape} from {file_path}")
        return frames, actions

    def _pick_indices(self, episode_len: int) -> Tuple[List[int], int, int]:
        if episode_len <= 0:
            raise ValueError("episode_len must be positive.")

        if self.random_selection:
            horizon = self.segment_horizon
            eff_step = max(1, episode_len // horizon) if self.stepsize * horizon > episode_len else self.stepsize
            max_start = max(episode_len - eff_step * horizon + 1, 1)
            start = 0 if self.from_start else int(self.rng.integers(0, max_start))

            all_idx = list(range(start, min(start + eff_step * horizon, episode_len)))
            ctx_idx = all_idx[: eff_step * self.context_length : eff_step]
            after_idx = all_idx[eff_step * self.context_length :]

            num_after = min(len(after_idx), max(self.segment_length - self.context_length, 0))
            if num_after > 0:
                choose = np.sort(self.rng.choice(len(after_idx), size=num_after, replace=False))
                sampled_after = [after_idx[int(i)] for i in choose]
            else:
                sampled_after = []
            indices = ctx_idx + sampled_after
        else:
            eff_step = (
                max(1, episode_len // self.segment_length)
                if self.stepsize * self.segment_length > episode_len
                else self.stepsize
            )
            max_start = max(episode_len - eff_step * self.segment_length + 1, 1)
            start = 0 if self.from_start else int(self.rng.integers(0, max_start))
            indices = list(range(start, min(start + eff_step * self.segment_length, episode_len), eff_step))

        if not indices:
            indices = [0]

        while len(indices) < self.segment_length:
            indices.append(indices[-1])

        if self.random_ctx_frame:
            low = max(start - 15, 0)
            high = start + 1
            ctx_index = int(self.rng.integers(low, high))
            indices = [ctx_index] + indices

        return indices, start, eff_step

    def _resize(self, images_tchw: Any) -> Any:
        # images_tchw: [T, C, H, W], uint8 range converted to [0,1]
        _, F = _require_torch()
        return F.interpolate(
            images_tchw,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def sample_one(self) -> Tuple[Any, Optional[Any], SegmentMeta]:
        torch, _ = _require_torch()
        file_path = self.files[int(self.rng.integers(0, len(self.files)))]
        frames, actions = self._load_episode(file_path)
        indices, start, eff_step = self._pick_indices(len(frames))

        sel_frames = frames[indices]  # [T, H, W, C]
        pixels = torch.from_numpy(sel_frames).permute(0, 3, 1, 2).float() / 255.0
        pixels = self._resize(pixels)

        sel_actions = None
        if self.load_action:
            if actions is None:
                raise KeyError(f"`action` key not found in {file_path} but `load_action=True`.")
            sel_actions = torch.from_numpy(actions[indices]).float()

        meta = SegmentMeta(
            file_path=file_path,
            episode_length=int(len(frames)),
            start=int(start),
            stepsize=int(eff_step),
            frame_indices=[int(i) for i in indices],
        )
        return pixels, sel_actions, meta

    def sample_batch(self, batch_size: int) -> Tuple[Any, Optional[Any], List[SegmentMeta]]:
        torch, _ = _require_torch()
        pixels_list: List[Any] = []
        actions_list: List[Any] = []
        meta_list: List[SegmentMeta] = []

        for _ in range(batch_size):
            pixels, actions, meta = self.sample_one()
            pixels_list.append(pixels)
            if actions is not None:
                actions_list.append(actions)
            meta_list.append(meta)

        pixel_batch = torch.stack(pixels_list, dim=0)  # [B, T, C, H, W]
        action_batch = torch.stack(actions_list, dim=0) if actions_list else None
        return pixel_batch, action_batch, meta_list


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone loader script that mimics RLVR simple_dataloader behavior.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path that contains {dataset_name}/*.npz")
    parser.add_argument("--dataset_name", type=str, default="fractal20220817_data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--segment_length", type=int, default=5)
    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument("--stepsize", type=int, default=1)
    parser.add_argument("--random_selection", action="store_true", help="Match random_selection=True branch.")
    parser.add_argument("--segment_horizon", type=int, default=None)
    parser.add_argument("--from_start", action="store_true")
    parser.add_argument("--random_ctx_frame", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--no_action", action="store_true", help="Disable loading action.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _summarize_batch(
    batch_idx: int,
    pixels: Any,
    actions: Optional[Any],
    metas: Sequence[SegmentMeta],
) -> dict:
    init = extract_initial_image(pixels)
    goal = extract_goal_image(pixels)
    return {
        "batch_idx": batch_idx,
        "pixels_shape": list(pixels.shape),
        "actions_shape": list(actions.shape) if actions is not None else None,
        "init_shape": list(init.shape),
        "goal_shape": list(goal.shape),
        "files": [m.file_path for m in metas],
        "segment_meta": [
            {
                "episode_length": m.episode_length,
                "start": m.start,
                "stepsize": m.stepsize,
                "frame_indices": m.frame_indices,
            }
            for m in metas
        ],
    }


def main():
    args = _parse_args()

    loader = RLVRLikeLoader(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        segment_length=args.segment_length,
        context_length=args.context_length,
        stepsize=args.stepsize,
        random_selection=bool(args.random_selection),
        segment_horizon=args.segment_horizon,
        from_start=bool(args.from_start),
        random_ctx_frame=bool(args.random_ctx_frame),
        image_size=args.image_size,
        load_action=not args.no_action,
        seed=args.seed,
    )

    header = {
        "mode": "rlvr_like_loader",
        "dataset_path": os.path.abspath(args.dataset_path),
        "dataset_name": args.dataset_name,
        "display_key": loader.display_key,
        "split": args.split,
        "num_files": len(loader.files),
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "segment_length": args.segment_length,
        "context_length": args.context_length,
        "stepsize": args.stepsize,
        "random_selection": bool(args.random_selection),
        "segment_horizon": args.segment_horizon,
        "from_start": bool(args.from_start),
        "random_ctx_frame": bool(args.random_ctx_frame),
        "image_size": args.image_size,
        "load_action": not args.no_action,
        "seed": args.seed,
    }

    out = {"config": header, "batches": []}
    for i in range(args.num_batches):
        pixels, actions, metas = loader.sample_batch(args.batch_size)
        out["batches"].append(_summarize_batch(i, pixels, actions, metas))

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

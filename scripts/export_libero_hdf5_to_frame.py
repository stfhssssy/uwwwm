#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


def _sorted_hdf5_files(input_roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in input_roots:
        files.extend(sorted(Path(root).glob("*.hdf5")))
    files = sorted([p.resolve() for p in files])
    return files


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    keys = list(data_group.keys())

    def sort_key(k: str) -> tuple[int, str]:
        # demo_12 -> 12
        try:
            return int(k.split("_")[-1]), k
        except Exception:
            return 10**9, k

    return sorted(keys, key=sort_key)


def _export_single_hdf5(
    hdf5_path: str,
    output_root: str,
    views: list[str],
    image_ext: str,
    skip_existing_demo: bool,
) -> dict[str, Any]:
    src = Path(hdf5_path).resolve()
    split = src.parent.name
    task_name = src.stem.replace("_demo", "")
    dst_task_dir = Path(output_root).resolve() / split / task_name

    saved_frames = 0
    skipped_frames = 0
    demos_total = 0
    demos_done = 0
    missing_views: dict[str, int] = {v: 0 for v in views}

    t0 = time.time()
    with h5py.File(src, "r") as f:
        data_group = f["data"]
        demo_keys = _sorted_demo_keys(data_group)
        demos_total = len(demo_keys)

        for demo_key in demo_keys:
            obs_group = data_group[demo_key]["obs"]
            demo_has_all_views = all(v in obs_group for v in views)
            if not demo_has_all_views:
                for v in views:
                    if v not in obs_group:
                        missing_views[v] += 1
                continue

            # Use first view as reference length.
            t = int(obs_group[views[0]].shape[0])

            # Fast skip when all view folders already contain exactly t frames.
            if skip_existing_demo:
                done = True
                for v in views:
                    vdir = dst_task_dir / demo_key / v
                    if not vdir.exists():
                        done = False
                        break
                    if len(list(vdir.glob(f"*.{image_ext}"))) != t:
                        done = False
                        break
                if done:
                    skipped_frames += t * len(views)
                    demos_done += 1
                    continue

            for v in views:
                arr = obs_group[v]
                out_dir = dst_task_dir / demo_key / v
                out_dir.mkdir(parents=True, exist_ok=True)

                # Remove trailing frames if interrupted and resumed with fewer steps.
                if skip_existing_demo:
                    existing = sorted(out_dir.glob(f"*.{image_ext}"))
                    if len(existing) > t:
                        for p in existing[t:]:
                            p.unlink()

                for i in range(t):
                    out_path = out_dir / f"{i:06d}.{image_ext}"
                    if skip_existing_demo and out_path.exists():
                        skipped_frames += 1
                        continue
                    frame = np.asarray(arr[i], dtype=np.uint8)
                    Image.fromarray(frame).save(out_path)
                    saved_frames += 1

            demos_done += 1

    return {
        "file": str(src),
        "split": split,
        "task": task_name,
        "demos_total": demos_total,
        "demos_done": demos_done,
        "saved_frames": saved_frames,
        "skipped_frames": skipped_frames,
        "missing_views": missing_views,
        "elapsed_sec": round(time.time() - t0, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LIBERO HDF5 trajectories to frame folders for PROGRESSOR/Video2Reward."
    )
    parser.add_argument(
        "--input-roots",
        nargs="+",
        default=[
            "/mnt/hdd/ssy/libero_data/datasets/libero_10",
            "/mnt/hdd/ssy/libero_data/datasets/libero_90",
        ],
        help="One or more directories that contain *.hdf5.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/hdd/ssy/frame",
        help="Output root. Example: /mnt/hdd/ssy/frame",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["agentview_rgb", "eye_in_hand_rgb"],
        help="Observation image keys to export from each demo.",
    )
    parser.add_argument(
        "--image-ext",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image extension.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers per hdf5 file.",
    )
    parser.add_argument(
        "--no-skip-existing-demo",
        action="store_true",
        help="If set, rewrite existing frames instead of resuming.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    files = _sorted_hdf5_files(args.input_roots)
    if len(files) == 0:
        raise RuntimeError(f"No hdf5 files found under: {args.input_roots}")

    skip_existing_demo = not bool(args.no_skip_existing_demo)
    workers = max(1, int(args.workers))

    print(f"[start] files={len(files)} output_root={output_root} workers={workers}")
    print(f"[views] {args.views}")
    print(f"[resume] skip_existing_demo={skip_existing_demo}")

    total_saved = 0
    total_skipped = 0
    results: list[dict[str, Any]] = []
    t0 = time.time()

    with mp.get_context("spawn").Pool(processes=workers) as pool:
        async_results = [
            pool.apply_async(
                _export_single_hdf5,
                kwds={
                    "hdf5_path": str(fp),
                    "output_root": str(output_root),
                    "views": list(args.views),
                    "image_ext": str(args.image_ext),
                    "skip_existing_demo": skip_existing_demo,
                },
            )
            for fp in files
        ]

        for idx, ar in enumerate(async_results, start=1):
            r = ar.get()
            results.append(r)
            total_saved += int(r["saved_frames"])
            total_skipped += int(r["skipped_frames"])
            print(
                f"[{idx}/{len(files)}] {Path(r['file']).name} "
                f"demos={r['demos_done']}/{r['demos_total']} "
                f"saved={r['saved_frames']} skipped={r['skipped_frames']} "
                f"elapsed={r['elapsed_sec']}s"
            )

    elapsed = round(time.time() - t0, 3)
    summary = {
        "input_roots": args.input_roots,
        "output_root": str(output_root),
        "views": args.views,
        "image_ext": args.image_ext,
        "workers": workers,
        "skip_existing_demo": skip_existing_demo,
        "num_files": len(files),
        "total_saved_frames": total_saved,
        "total_skipped_frames": total_skipped,
        "elapsed_sec": elapsed,
        "results": results,
    }

    summary_path = output_root / "export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] total_saved={total_saved} total_skipped={total_skipped} elapsed={elapsed}s")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()

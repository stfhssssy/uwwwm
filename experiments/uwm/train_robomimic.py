import h5py
import hydra
import inspect
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from diffusers.optimization import get_scheduler
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from tqdm import trange, tqdm

from datasets.utils.file_utils import glob_all
from datasets.utils.loader import make_distributed_data_loader
from environments.robomimic import make_robomimic_env
from experiments.utils import set_seed, init_wandb, init_distributed, is_main_process
from experiments.uwm.train import (
    train_one_step,
    maybe_resume_checkpoint,
    maybe_evaluate,
    maybe_save_checkpoint,
)


def sanitize_task_name(name: str) -> str:
    task_name = name.replace(".hdf5", "").replace("_demo", "")
    task_name = task_name.replace("/", "_").replace(" ", "_")
    return task_name


def get_rgb_obs_keys(shape_meta: dict) -> list[str]:
    return [k for k, v in shape_meta["obs"].items() if v["type"] == "rgb"]


def build_goal_image_pool(
    shape_meta: dict, hdf5_paths: list[str], flip_rgb: bool = False
) -> dict[str, torch.Tensor]:
    rgb_keys = get_rgb_obs_keys(shape_meta)
    goal_pool = {k: [] for k in rgb_keys}

    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path) as f:
            demos = f["data"]
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]
                for key in rgb_keys:
                    goal_frame = demo["obs"][key][-1:]
                    if flip_rgb:
                        goal_frame = goal_frame[:, ::-1].copy()
                    goal_pool[key].append(goal_frame)

    if len(goal_pool[rgb_keys[0]]) == 0:
        raise RuntimeError("No goal images found when building goal pool.")

    goal_pool = {
        key: torch.from_numpy(np.stack(goal_frames, axis=0))
        for key, goal_frames in goal_pool.items()
    }
    return goal_pool


def get_rollout_goal(
    goal_pool: dict[str, torch.Tensor] | None, rollout_index: int, device
) -> dict[str, torch.Tensor] | None:
    if goal_pool is None:
        return None
    num_goals = next(iter(goal_pool.values())).shape[0]
    goal_idx = rollout_index % num_goals
    return {key: value[goal_idx : goal_idx + 1].to(device) for key, value in goal_pool.items()}


def get_rollout_tasks(config) -> list[dict]:
    hdf5_paths = glob_all(config.dataset.hdf5_path_globs)
    if len(hdf5_paths) == 0:
        raise RuntimeError("No hdf5 files found for rollout evaluation.")

    # Treat each dataset file as one eval task.
    task_specs = []
    for hdf5_path in hdf5_paths:
        task_name = sanitize_task_name(hdf5_path.split("/")[-1])
        task_specs.append(
            {
                "task_name": task_name,
                "dataset_path": hdf5_path,
                "hdf5_paths": [hdf5_path],
            }
        )
    return task_specs


def collect_rollout(config, model, device):
    model.eval()
    model = getattr(model, "module", model)  # unwrap DDP
    supports_goal_obs = "goal_obs_dict" in inspect.signature(model.sample).parameters

    rollout_stats = {}
    task_specs = get_rollout_tasks(config)
    rollouts_per_task = config.num_rollouts

    for task_spec in task_specs:
        task_name = task_spec["task_name"]

        # Create task-specific eval environment
        env = make_robomimic_env(
            dataset_name=config.dataset.name,
            dataset_path=task_spec["dataset_path"],
            shape_meta=config.dataset.shape_meta,
            obs_horizon=model.obs_encoder.num_frames,
            max_episode_length=config.rollout_length,
            record=True,
        )

        goal_pool = None
        if getattr(model.obs_encoder, "use_goal_image_cond", False):
            goal_pool = build_goal_image_pool(
                shape_meta=config.dataset.shape_meta,
                hdf5_paths=task_spec["hdf5_paths"],
                flip_rgb=getattr(config.dataset, "flip_rgb", False),
            )
            print(
                f"Loaded {next(iter(goal_pool.values())).shape[0]} goal images for task {task_name}."
            )

        successes = []
        for e in trange(
            rollouts_per_task,
            desc=f"Collecting {task_name}",
            disable=not is_main_process(),
        ):
            env.seed(e)
            obs = env.reset()
            goal_obs_tensor = get_rollout_goal(goal_pool, e, device)
            done = False
            while not done:
                obs_tensor = {
                    k: torch.tensor(v, device=device)[None] for k, v in obs.items()
                }

                # Keep compatibility with models that do not accept goal_obs_dict.
                if supports_goal_obs:
                    action = model.sample(obs_tensor, goal_obs_dict=goal_obs_tensor)
                else:
                    action = model.sample(obs_tensor)
                action = action[0].cpu().numpy()

                # Step environment
                obs, reward, done, info = env.step(action)
            successes.append(info["success"])

        success_rate = sum(successes) / len(successes)
        rollout_stats[task_name] = {
            "success_rate": success_rate,
            "video": env.get_video(),
        }
        env.close()

    return rollout_stats


def maybe_collect_rollout(config, step, model, device):
    """Collect rollouts on the main process if it's the correct step."""
    # Skip rollout rollection for pretraining
    if "libero_90" in config.dataset.name:
        return

    if is_main_process() and (
        step % config.rollout_every == 0 or step == (config.num_steps - 1)
    ):
        rollout_stats = collect_rollout(config, model, device)
        log_data = {}
        success_rates = []
        first_video = None
        for task_name, task_stats in rollout_stats.items():
            success_rate = task_stats["success_rate"]
            video_np = task_stats["video"].transpose(0, 3, 1, 2)[None]
            success_rates.append(success_rate)
            print(f"Step: {step}, task: {task_name}, success rate: {success_rate}")
            log_data[f"rollout/{task_name}/success_rate"] = success_rate
            log_data[f"rollout/{task_name}/video"] = wandb.Video(video_np, fps=10)
            if first_video is None:
                first_video = video_np

        # Keep the old aggregate key for backward compatibility with dashboards.
        log_data["rollout/success_rate"] = float(np.mean(success_rates))
        if first_video is not None:
            log_data["rollout/video"] = wandb.Video(first_video, fps=10)
        wandb.log(log_data)
    dist.barrier()


def train(rank, world_size, config):
    # Set global seed
    set_seed(config.seed * world_size + rank)

    # Initialize distributed training
    init_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Initialize WANDB
    if is_main_process():
        init_wandb(config, job_type="train")

    # Create dataset and loader
    train_set, val_set = instantiate(config.dataset)
    train_loader, val_loader = make_distributed_data_loader(
        train_set, val_set, config.batch_size, rank, world_size
    )

    # Create model
    model = instantiate(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # Load pretrained model
    if config.pretrain_checkpoint_path:
        ckpt = torch.load(config.pretrain_checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(
            f"Loaded pretraining checkpoint {config.pretrain_checkpoint_path}, step: {ckpt['step']}"
        )

        # Use a smaller learning rate for the image encoder
        encoder_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("obs_encoder.img_encoder"):
                encoder_params.append(param)
            else:
                other_params.append(param)

        # Define learning rates
        base_lr = config.optimizer.lr
        encoder_lr = base_lr * 0.05
        wd = config.optimizer.weight_decay

        # Construct optimizer with custom parameter groups
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": encoder_lr, "weight_decay": wd},
                {"params": other_params, "lr": base_lr, "weight_decay": wd},
            ],
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
        )
        scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)

    # Resume from checkpoint
    step = maybe_resume_checkpoint(config, model, optimizer, scheduler, scaler)
    epoch = step // len(train_loader)

    # Wrap model with DDP
    model = DistributedDataParallel(model, device_ids=[rank], static_graph=True)

    # Training loop
    pbar = tqdm(
        total=config.num_steps,
        initial=step,
        desc="Training",
        disable=not is_main_process(),
    )
    while step < config.num_steps:
        # Set epoch for distributed sampler to shuffle indices
        train_loader.sampler.set_epoch(epoch)

        # Train for one epoch
        for batch in train_loader:
            # --- Training step ---
            loss, info = train_one_step(
                config, model, optimizer, scheduler, scaler, batch, device
            )

            # --- Logging ---
            if is_main_process():
                pbar.set_description(f"step: {step}, loss: {loss.item():.4f}")
                wandb.log({f"train/{k}": v for k, v in info.items()})

            # --- Evaluate if needed ---
            maybe_evaluate(config, step, model, val_loader, device)

            # ---Collect environment rollouts if needed ---
            maybe_collect_rollout(config, step, model, device)

            # --- Save checkpoint if needed ---
            maybe_save_checkpoint(config, step, model, optimizer, scheduler, scaler)

            step += 1
            pbar.update(1)
            if step >= config.num_steps:
                break

        epoch += 1


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_uwm_robomimic.yaml",
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    # Spawn processes
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

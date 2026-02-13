import os

import h5py
import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from datasets.utils.file_utils import glob_all
from environments.robomimic import make_robomimic_env
from experiments.utils import set_seed, is_main_process


def get_rgb_obs_keys(shape_meta: dict) -> list[str]:
    return [k for k, v in shape_meta["obs"].items() if v["type"] == "rgb"]


def build_goal_image_pool(config) -> dict[str, torch.Tensor]:
    """
    Build a pool of episode-final RGB frames for the current task.
    Returns per-key tensors of shape (N, 1, H, W, C), where N is the number of demos.
    """
    rgb_keys = get_rgb_obs_keys(config.dataset.shape_meta)
    goal_pool = {k: [] for k in rgb_keys}

    hdf5_paths = glob_all(config.dataset.hdf5_path_globs)
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path) as f:
            demos = f["data"]
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]
                for key in rgb_keys:
                    goal_frame = demo["obs"][key][-1:]
                    if getattr(config.dataset, "flip_rgb", False):
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


def collect_rollout(config, model, device):
    model.eval()
    model = getattr(model, "module", model)  # unwrap DDP

    # Create eval environment
    assert isinstance(config.dataset.hdf5_path_globs, str)
    env = make_robomimic_env(
        dataset_name=config.dataset.name,
        dataset_path=config.dataset.hdf5_path_globs,
        shape_meta=config.dataset.shape_meta,
        obs_horizon=model.obs_encoder.num_frames,
        max_episode_length=config.rollout_length,
        record=True,
    )
    goal_pool = None
    if model.obs_encoder.use_goal_image_cond:
        goal_pool = build_goal_image_pool(config)
        print(f"Loaded {next(iter(goal_pool.values())).shape[0]} goal images for rollout.")

    # Collect rollouts
    video_dir = os.path.join(config.logdir, "videos")
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    successes = []
    for e in trange(
        config.num_rollouts, desc="Collecting rollouts", disable=not is_main_process()
    ):
        env.seed(e)
        obs = env.reset()
        goal_obs_tensor = get_rollout_goal(goal_pool, e, device)
        done = False
        while not done:
            obs_tensor = {
                k: torch.tensor(v, device=device)[None] for k, v in obs.items()
            }

            # Sample action from model
            action = model.sample(obs_tensor, goal_obs_dict=goal_obs_tensor)[0].cpu().numpy()

            # Step environment
            obs, reward, done, info = env.step(action)
        successes.append(info["success"])
        video = env.get_video()
        imageio.mimwrite(os.path.join(video_dir, f"{e}.mp4"), video, fps=30)
        print(
            f"Episode {e} success: {info['success']}, cumulative: {np.mean(successes):.2f}"
        )

    # Compute success rate
    success_rate = sum(successes) / len(successes)
    return success_rate


def process_batch(
    batch,
    obs_horizon,
    action_horizon,
    device,
):
    action_start = obs_horizon - 1
    action_end = action_start + action_horizon
    curr_obs = {k: v[:, : action_start + 1].to(device) for k, v in batch["obs"].items()}
    next_obs = {k: v[:, action_end:].to(device) for k, v in batch["obs"].items()}
    actions = batch["action"][:, action_start:action_end].to(device)

    # Add language tokens if present
    if "input_ids" in batch and "attention_mask" in batch:
        curr_obs["input_ids"] = batch["input_ids"].to(device)
        curr_obs["attention_mask"] = batch["attention_mask"].to(device)
    return curr_obs, next_obs, actions


def eval_inverse_dynamics(config, model, device, action_normalizer=None):
    model.eval()
    model = getattr(model, "module", model)  # unwrap DDP

    # Build dataset loaders for offline inverse dynamics eval
    train_set, val_set = instantiate(config.dataset)
    if action_normalizer is not None:
        train_set.action_normalizer = action_normalizer
        val_set.action_normalizer = action_normalizer

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, "eval_num_workers", 4),
        pin_memory=True,
        drop_last=False,
        persistent_workers=getattr(config, "eval_num_workers", 4) > 0,
    )

    if action_normalizer is not None:
        action_scale = torch.tensor(action_normalizer.scale[None], device=device)
        action_offset = torch.tensor(action_normalizer.offset[None], device=device)

        def unnormalize(actions):
            return actions * action_scale + action_offset

    else:

        def unnormalize(actions):
            return actions

    total_mse = 0.0
    num_batches = 0
    for batch in tqdm(val_loader, desc="Evaluating inverse dynamics"):
        curr_obs_dict, next_obs_dict, action = process_batch(
            batch, config.model.obs_encoder.num_frames, config.model.action_len, device
        )
        with torch.no_grad():
            action_hat = model.sample_inverse_dynamics(curr_obs_dict, next_obs_dict)
        action = unnormalize(action)
        action_hat = unnormalize(action_hat)
        total_mse += F.mse_loss(action_hat, action).item()
        num_batches += 1

    avg_mse = total_mse / max(num_batches, 1)
    print(f"Inverse dynamics action MSE: {avg_mse:.6f}")
    return avg_mse


@torch.no_grad()
def sample_inverse_dynamics_from_latent(model, obs_dict, next_obs_latent):
    obs_feat = model.obs_encoder.encode_curr_obs(obs_dict)
    action_sample = torch.randn(
        (obs_feat.shape[0],) + model.action_shape, device=obs_feat.device
    )

    model.noise_scheduler.set_timesteps(model.num_inference_steps)
    next_obs_t = model.noise_scheduler.timesteps[-1]
    for action_t in model.noise_scheduler.timesteps:
        action_noise_pred, _ = model.noise_pred_net(
            obs_feat, action_sample, action_t, next_obs_latent, next_obs_t
        )
        action_sample = model.noise_scheduler.step(
            action_noise_pred, action_t, action_sample
        ).prev_sample
    return action_sample


def collect_rollout_with_future_obs(config, model, device, action_normalizer=None):
    model.eval()
    model = getattr(model, "module", model)  # unwrap DDP

    # Create eval environment
    assert isinstance(config.dataset.hdf5_path_globs, str)
    env = make_robomimic_env(
        dataset_name=config.dataset.name,
        dataset_path=config.dataset.hdf5_path_globs,
        shape_meta=config.dataset.shape_meta,
        obs_horizon=model.obs_encoder.num_frames,
        max_episode_length=config.rollout_length,
        record=True,
    )

    if not hasattr(env, "reset_to_state"):
        raise RuntimeError("Environment does not support reset_to_state for cheating eval.")

    if action_normalizer is not None:
        action_scale = torch.tensor(action_normalizer.scale[None], device=device)
        action_offset = torch.tensor(action_normalizer.offset[None], device=device)

        def unnormalize(actions):
            return actions * action_scale + action_offset

    else:

        def unnormalize(actions):
            return actions

    # Collect rollouts using future observations from the dataset
    obs_keys = list(config.dataset.shape_meta["obs"].keys())
    obs_horizon = model.obs_encoder.num_frames
    action_horizon = model.action_len
    rgb_keys = [
        k for k, v in config.dataset.shape_meta["obs"].items() if v["type"] == "rgb"
    ]
    successes = []

    hdf5_paths = glob_all(config.dataset.hdf5_path_globs)
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path) as f:
            demos = f["data"]
            for i in trange(len(demos), desc="Cheating rollouts"):
                if len(successes) >= config.num_rollouts:
                    break
                demo = demos[f"demo_{i}"]
                if "states" not in demo:
                    raise RuntimeError("Dataset does not include 'states' for reset.")

                obs = env.reset_to_state(demo["states"][0])
                action_start = obs_horizon - 1
                if action_start > 0:
                    warm_actions = demo["actions"][:action_start]
                    if len(warm_actions) < action_start:
                        continue
                    obs, _, done, info = env.step(warm_actions)
                    if done:
                        successes.append(info["success"])
                        if len(successes) >= config.num_rollouts:
                            break
                        continue

                next_index = action_start + action_horizon
                done = False
                while next_index + obs_horizon <= len(demo["actions"]) and not done:
                    next_obs = {}
                    for key in obs_keys:
                        obs_seq = demo["obs"][key][
                            next_index : next_index + obs_horizon
                        ]
                        if getattr(config.dataset, "flip_rgb", False) and key in rgb_keys:
                            obs_seq = obs_seq[:, ::-1].copy()
                        next_obs[key] = obs_seq

                    obs_tensor = {
                        k: torch.tensor(v, device=device)[None] for k, v in obs.items()
                    }
                    next_obs_tensor = {
                        k: torch.tensor(v, device=device)[None]
                        for k, v in next_obs.items()
                    }

                    action = model.sample_inverse_dynamics(obs_tensor, next_obs_tensor)
                    action = unnormalize(action)[0].cpu().numpy()
                    obs, _, done, info = env.step(action)
                    next_index += action_horizon

                successes.append(info["success"])
                if len(successes) >= config.num_rollouts:
                    break

    if len(successes) == 0:
        raise RuntimeError("No rollouts collected for cheating eval.")
    success_rate = sum(successes) / len(successes)
    return success_rate


def collect_rollout_with_predicted_future_obs(
    config, model, device, action_normalizer=None
):
    model.eval()
    model = getattr(model, "module", model)  # unwrap DDP

    # Create eval environment
    assert isinstance(config.dataset.hdf5_path_globs, str)
    env = make_robomimic_env(
        dataset_name=config.dataset.name,
        dataset_path=config.dataset.hdf5_path_globs,
        shape_meta=config.dataset.shape_meta,
        obs_horizon=model.obs_encoder.num_frames,
        max_episode_length=config.rollout_length,
        record=True,
    )

    if action_normalizer is not None:
        action_scale = torch.tensor(action_normalizer.scale[None], device=device)
        action_offset = torch.tensor(action_normalizer.offset[None], device=device)

        def unnormalize(actions):
            return actions * action_scale + action_offset

    else:

        def unnormalize(actions):
            return actions

    # Collect rollouts
    video_dir = os.path.join(config.logdir, "videos")
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    successes = []
    for e in trange(
        config.num_rollouts, desc="Collecting rollouts", disable=not is_main_process()
    ):
        env.seed(e)
        obs = env.reset()
        done = False
        while not done:
            obs_tensor = {
                k: torch.tensor(v, device=device)[None] for k, v in obs.items()
            }

            # Predict future obs, then condition inverse dynamics on it
            next_obs_latent = model.sample_marginal_next_obs(obs_tensor)
            action = sample_inverse_dynamics_from_latent(model, obs_tensor, next_obs_latent)
            action = unnormalize(action)[0].cpu().numpy()

            # Step environment
            obs, reward, done, info = env.step(action)
        successes.append(info["success"])
        video = env.get_video()
        imageio.mimwrite(os.path.join(video_dir, f"{e}.mp4"), video, fps=30)
        print(
            f"Episode {e} success: {info['success']}, cumulative: {np.mean(successes):.2f}"
        )

    # Compute success rate
    success_rate = sum(successes) / len(successes)
    return success_rate


def maybe_collect_rollout(config, step, model, device):
    """Collect rollouts on the main process if it's the correct step."""
    # Skip rollout rollection for pretraining
    if "libero_90" in config.dataset.name:
        return

    if is_main_process() and (
        step % config.rollout_every == 0 or step == (config.num_steps - 1)
    ):
        success_rate = collect_rollout(config, model, device)
        print(f"Step: {step} success rate: {success_rate}")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_uwm_robomimic.yaml",
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)
    set_seed(0)
    device = torch.device(f"cuda:0")

    # Create model
    model = instantiate(config.model).to(device)

    # Load model weights only (avoid optimizer param-group mismatches in eval)
    ckpt_path = os.path.join(config.logdir, "models.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint {ckpt_path}, step: {ckpt.get('step', 'unknown')}")
    if getattr(config, "eval_use_predicted_future_obs", False):
        success_rate = collect_rollout_with_predicted_future_obs(
            config, model, device, action_normalizer=ckpt.get("action_normalizer")
        )
        print(f"Predicted-future success rate: {success_rate}")
    elif getattr(config, "eval_use_future_obs", False):
        if getattr(config, "eval_future_obs_in_env", False):
            success_rate = collect_rollout_with_future_obs(
                config, model, device, action_normalizer=ckpt.get("action_normalizer")
            )
            print(f"Cheating success rate: {success_rate}")
        else:
            eval_inverse_dynamics(
                config, model, device, action_normalizer=ckpt.get("action_normalizer")
            )
    else:
        maybe_collect_rollout(config, 0, model, device)


if __name__ == "__main__":
    main()

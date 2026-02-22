import copy
from collections import deque

import numpy as np


class RoboMimicEnvWrapper:
    def __init__(
        self,
        env,
        obs_keys,
        obs_horizon,
        max_episode_length,
        record=False,
        render_size=(224, 224),
        terminate_on_success=True,
    ):
        self.env = env
        self.obs_keys = obs_keys
        self.obs_buffer = deque(maxlen=obs_horizon)

        self._max_episode_length = max_episode_length
        self._elapsed_steps = None

        self.record = record
        self.render_size = render_size
        self.terminate_on_success = terminate_on_success
        if record:
            self.video_buffer = deque()

    def _is_success(self):
        return self.env.is_success()["task"]

    @staticmethod
    def _clone_obs_dict(obs):
        return {k: np.array(v, copy=True) for k, v in obs.items()}

    def _clone_stacked_obs(self, stacked_obs):
        return {k: np.array(v, copy=True) for k, v in stacked_obs.items()}

    def _get_obs_buffer_list(self):
        return [self._clone_obs_dict(obs) for obs in self.obs_buffer]

    def _get_obs(self):
        # Return a dictionary of stacked observations
        stacked_obs = {}
        for key in self.obs_keys:
            stacked_obs[key] = np.stack([obs[key] for obs in self.obs_buffer])
        return stacked_obs

    def _normalize_actions(self, actions):
        if hasattr(actions, "detach") and hasattr(actions, "cpu") and hasattr(actions, "numpy"):
            actions_arr = actions.detach().cpu().numpy()
        elif isinstance(actions, np.ndarray):
            actions_arr = actions
        else:
            actions_arr = np.asarray(actions)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr[None]
        return actions_arr

    def get_sim_state(self):
        if hasattr(self.env, "get_state"):
            return self.env.get_state()
        if hasattr(self.env, "sim") and hasattr(self.env.sim, "get_state"):
            return self.env.sim.get_state()
        raise RuntimeError("Underlying environment does not expose get_state().")

    def get_snapshot(self):
        return {
            "sim_state": copy.deepcopy(self.get_sim_state()),
            "obs_buffer": self._get_obs_buffer_list(),
            "elapsed_steps": int(self._elapsed_steps if self._elapsed_steps is not None else 0),
        }

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        # Clear buffers
        self.obs_buffer.clear()
        if self.record:
            self.video_buffer.clear()

        # Reset environment
        obs = self.env.reset()
        self._elapsed_steps = 0

        # Pad observation buffer
        for _ in range(self.obs_buffer.maxlen):
            self.obs_buffer.append(obs)

        return self._get_obs()

    def reset_to_state(self, state):
        self.seed(0)
        self.obs_buffer.clear()
        if self.record:
            self.video_buffer.clear()
        self._elapsed_steps = 0

        # Prefer robomimic-style reset_to when available
        if hasattr(self.env, "reset_to"):
            obs = self.env.reset_to(state)
        else:
            obs = self.env.reset()
            if hasattr(self.env, "set_state"):
                self.env.set_state(state)
                if hasattr(self.env, "_get_observations"):
                    obs = self.env._get_observations()
                elif hasattr(self.env, "get_observation"):
                    obs = self.env.get_observation()

        # Pad observation buffer
        for _ in range(self.obs_buffer.maxlen):
            self.obs_buffer.append(obs)
        return self._get_obs()

    def reset_to_snapshot(self, snapshot):
        obs = self.reset_to_state(snapshot["sim_state"])
        obs_buffer = snapshot.get("obs_buffer", None)
        if obs_buffer is not None and len(obs_buffer) > 0:
            self.obs_buffer.clear()
            for frame_obs in obs_buffer[-self.obs_buffer.maxlen :]:
                self.obs_buffer.append(self._clone_obs_dict(frame_obs))
            while len(self.obs_buffer) < self.obs_buffer.maxlen and len(obs_buffer) > 0:
                self.obs_buffer.append(self._clone_obs_dict(obs_buffer[0]))
            obs = self._get_obs()
        self._elapsed_steps = int(snapshot.get("elapsed_steps", 0))
        return obs

    def step_with_trace(
        self,
        actions,
        terminate_on_success=None,
        capture_state_trace=False,
        capture_obs_trace=False,
        return_final_snapshot=True,
    ):
        # Roll out a sequence of actions in the environment with per-step trace.
        if terminate_on_success is None:
            terminate_on_success = self.terminate_on_success
        actions = self._normalize_actions(actions)
        if self._elapsed_steps is None:
            self._elapsed_steps = 0
        total_reward = 0
        done = False
        info = {}
        steps_executed = 0
        done_reason = "action_horizon_reached"
        state_trace = [] if capture_state_trace else None
        obs_trace = [] if capture_obs_trace else None

        for action in actions:
            # Step environment
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self.obs_buffer.append(obs)
            if self.record:
                self.video_buffer.append(self.render())
            steps_executed += 1

            # Store success info
            info["success"] = self._is_success()

            # Terminate on success
            if terminate_on_success and info["success"]:
                done = True
                done_reason = "success"
            elif done:
                done_reason = "env_done"

            # Terminate if max episode length is reached
            self._elapsed_steps += 1
            if self._elapsed_steps >= self._max_episode_length:
                info["truncated"] = not done
                done = True
                done_reason = "max_episode_length"

            if capture_state_trace:
                try:
                    state_trace.append(self.get_sim_state())
                except RuntimeError:
                    state_trace.append(None)
            if capture_obs_trace:
                obs_trace.append(self._clone_stacked_obs(self._get_obs()))

            if done:
                break

        result = {
            "obs": self._get_obs(),
            "reward": total_reward,
            "done": done,
            "info": info,
            "steps_executed": steps_executed,
            "done_reason": done_reason,
        }
        if capture_state_trace:
            result["state_trace"] = state_trace
        if capture_obs_trace:
            result["obs_trace"] = obs_trace
        if return_final_snapshot:
            try:
                result["final_snapshot"] = self.get_snapshot()
            except RuntimeError:
                result["final_snapshot"] = {
                    "sim_state": None,
                    "obs_buffer": self._get_obs_buffer_list(),
                    "elapsed_steps": int(self._elapsed_steps),
                }
        return result

    def step(self, actions):
        result = self.step_with_trace(
            actions,
            terminate_on_success=self.terminate_on_success,
            capture_state_trace=False,
            capture_obs_trace=False,
            return_final_snapshot=False,
        )
        return result["obs"], result["reward"], result["done"], result["info"]

    def render(self):
        return self.env.render(
            mode="rgb_array",
            width=self.render_size[0],
            height=self.render_size[1],
        )

    def get_video(self):
        if not self.record:
            raise ValueError("Video recording is disabled.")
        return np.stack(self.video_buffer)

    def close(self):
        self.env.close()


class LIBEROEnvWrapper(RoboMimicEnvWrapper):
    def __init__(
        self,
        env,
        obs_keys,
        obs_horizon,
        max_episode_length,
        record=False,
        render_size=(224, 224),
        terminate_on_success=True,
    ):
        super().__init__(
            env,
            obs_keys,
            obs_horizon,
            max_episode_length,
            record,
            render_size,
            terminate_on_success,
        )
        self.source_key_map = {
            "agentview_rgb": "agentview_image",
            "eye_in_hand_rgb": "robot0_eye_in_hand_image",
        }

    def reset_to_state(self, state):
        self.seed(0)
        self.reset()
        self.env.set_init_state(state)

        # Refresh obs buffer
        self.obs_buffer.clear()
        obs = self.env.env._get_observations()
        for _ in range(self.obs_buffer.maxlen):
            self.obs_buffer.append(obs)
        return self._get_obs()

    def get_sim_state(self):
        if hasattr(self.env, "get_sim_state"):
            return self.env.get_sim_state()
        if hasattr(self.env, "sim") and hasattr(self.env.sim, "get_state"):
            return self.env.sim.get_state()
        if hasattr(self.env, "env") and hasattr(self.env.env, "sim"):
            return self.env.env.sim.get_state()
        raise RuntimeError("Underlying LIBERO env does not expose get_sim_state().")

    def _is_success(self):
        return self.env.check_success()

    def _get_obs(self):
        # Return a dictionary of stacked observations
        stacked_obs = {}
        for key in self.obs_keys:
            source_key = self.source_key_map.get(key, key)
            stacked_obs[key] = np.stack([obs[source_key] for obs in self.obs_buffer])

        # Flip all image observations
        for key in self.obs_keys:
            if len(stacked_obs[key].shape) == 4:
                stacked_obs[key] = stacked_obs[key][:, ::-1].copy()
        return stacked_obs

    def render(self):
        img = self.env.env.sim.render(
            height=self.render_size[1],
            width=self.render_size[0],
            camera_name="frontview",
        )
        return img[::-1]

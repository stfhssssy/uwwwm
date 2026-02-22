import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import rearrange


from models.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from models.common.utils import SinusoidalPosEmb, init_weights
from .obs_encoder import UWMObservationEncoder


def _left_broadcast(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if t.ndim > len(shape):
        raise ValueError(f"cannot broadcast shape {tuple(t.shape)} to {shape}")
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).expand(shape)


def _normalize_timestep(
    timestep, batch_size: int, device: torch.device
) -> torch.LongTensor:
    if isinstance(timestep, int):
        return torch.full((batch_size,), timestep, dtype=torch.long, device=device)
    if not torch.is_tensor(timestep):
        return torch.full((batch_size,), int(timestep), dtype=torch.long, device=device)
    timestep = timestep.to(device=device, dtype=torch.long)
    if timestep.ndim == 0:
        return timestep.repeat(batch_size)
    if timestep.ndim == 1 and timestep.shape[0] == 1:
        return timestep.repeat(batch_size)
    if timestep.ndim == 1 and timestep.shape[0] == batch_size:
        return timestep
    raise ValueError(
        f"unsupported timestep shape {tuple(timestep.shape)} for batch size {batch_size}"
    )


def ddim_step_with_logprob(
    scheduler: DDIMScheduler,
    model_output: torch.Tensor,
    timestep,
    sample: torch.Tensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    prev_sample: torch.Tensor = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DDIM one-step transition with Gaussian log-probability.
    Returns:
      prev_sample: x_{t-1}
      log_prob: mean log p(x_{t-1} | x_t) over non-batch dims, shape (B,)
    """
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "scheduler.num_inference_steps is None; call set_timesteps before stepping"
        )

    batch_size = sample.shape[0]
    device = sample.device
    t = _normalize_timestep(timestep, batch_size=batch_size, device=device)

    step_size = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_t = t - step_size

    alphas_cumprod = scheduler.alphas_cumprod.to(device=device)
    t_safe = t.clamp(0, scheduler.config.num_train_timesteps - 1)
    prev_t_safe = prev_t.clamp(0, scheduler.config.num_train_timesteps - 1)
    alpha_prod_t = alphas_cumprod.gather(0, t_safe)
    alpha_prod_t_prev_table = alphas_cumprod.gather(0, prev_t_safe)

    if torch.is_tensor(scheduler.final_alpha_cumprod):
        final_alpha = scheduler.final_alpha_cumprod.to(device=device, dtype=alpha_prod_t.dtype)
    else:
        final_alpha = torch.tensor(
            scheduler.final_alpha_cumprod, device=device, dtype=alpha_prod_t.dtype
        )
    alpha_prod_t_prev = torch.where(
        prev_t >= 0, alpha_prod_t_prev_table, final_alpha.expand_as(alpha_prod_t_prev_table)
    )

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (
        1 - alpha_prod_t / alpha_prod_t_prev
    )
    variance = variance.clamp_min(0.0)

    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape)
    beta_prod_t = _left_broadcast(beta_prod_t, sample.shape)
    std_dev_t = _left_broadcast(eta * variance.sqrt(), sample.shape)

    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t.sqrt() * model_output
        ) / alpha_prod_t.sqrt()
        pred_epsilon = model_output
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t.sqrt() * pred_original_sample
        ) / beta_prod_t.sqrt()
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (
            alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        )
        pred_epsilon = (
            alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
        )
    else:
        raise ValueError(
            f"unsupported prediction_type={scheduler.config.prediction_type}"
        )

    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    if use_clipped_model_output:
        pred_epsilon = (
            sample - alpha_prod_t.sqrt() * pred_original_sample
        ) / beta_prod_t.sqrt()

    pred_sample_direction = (
        (1 - alpha_prod_t_prev - std_dev_t**2).clamp_min(0.0).sqrt() * pred_epsilon
    )
    prev_sample_mean = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

    if prev_sample is None:
        prev_sample = prev_sample_mean + std_dev_t * torch.randn_like(model_output)

    # Compute log-prob in fp32 for numerical stability, then average over non-batch dims.
    prev_sample_f = prev_sample.float()
    prev_sample_mean_f = prev_sample_mean.float()
    std_dev_t_f = std_dev_t.float().clamp_min(eps)
    log_prob = (
        -0.5 * ((prev_sample_f - prev_sample_mean_f) / std_dev_t_f).pow(2)
        - torch.log(std_dev_t_f)
        - 0.5 * math.log(2.0 * math.pi)
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample.to(sample.dtype), log_prob


class MultiViewVideoPatchifier(nn.Module):
    def __init__(
        self,
        num_views: int,
        input_shape: tuple[int, ...] = (8, 224, 224),
        patch_shape: tuple[int, ...] = (2, 8, 8),
        num_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_views = num_views
        iT, iH, iW = input_shape
        pT, pH, pW = patch_shape
        self.T, self.H, self.W = iT // pT, iH // pH, iW // pW
        self.pT, self.pH, self.pW = pT, pH, pW

        self.patch_encoder = nn.Conv3d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.patch_decoder = nn.Linear(embed_dim, num_chans * pT * pH * pW)

    def forward(self, imgs):
        return self.patchify(imgs)

    def patchify(self, imgs):
        imgs = rearrange(imgs, "b v c t h w -> (b v) c t h w")
        feats = self.patch_encoder(imgs)
        feats = rearrange(feats, "(b v) c t h w -> b (v t h w) c", v=self.num_views)
        return feats

    def unpatchify(self, feats):
        imgs = self.patch_decoder(feats)
        imgs = rearrange(
            imgs,
            "b (v t h w) (c pt ph pw) -> b v c (t pt) (h ph) (w pw)",
            v=self.num_views,
            t=self.T,
            h=self.H,
            w=self.W,
            pt=self.pT,
            ph=self.pH,
            pw=self.pW,
        )
        return imgs

    @property
    def num_patches(self):
        return self.num_views * self.T * self.H * self.W


class DualTimestepEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, mlp_ratio: float = 4.0):
        super().__init__()
        self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t1, t2):
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)
        temb = torch.cat([temb1, temb2], dim=-1)
        return self.proj(temb)


class DualNoisePredictionNet(nn.Module):
    def __init__(
        self,
        global_cond_dim: int,
        image_shape: tuple[int, ...],
        patch_shape: tuple[int, ...],
        num_chans: int,
        num_views: int,
        action_len: int,
        action_dim: int,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        num_registers: int = 8,
    ):
        super().__init__()
        # Observation encoder and decoder
        self.obs_patchifier = MultiViewVideoPatchifier(
            num_views=num_views,
            input_shape=image_shape,
            patch_shape=patch_shape,
            num_chans=num_chans,
            embed_dim=embed_dim,
        )
        obs_len = self.obs_patchifier.num_patches

        # Action encoder and decoder
        hidden_dim = int(max(action_dim, embed_dim) * mlp_ratio)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Timestep embedding
        self.timestep_embedding = DualTimestepEncoder(timestep_embed_dim)

        # Registers
        self.registers = nn.Parameter(
            torch.empty(1, num_registers, embed_dim).normal_(std=0.02)
        )

        # Positional embedding
        total_len = action_len + obs_len + num_registers
        self.pos_embed = nn.Parameter(
            torch.empty(1, total_len, embed_dim).normal_(std=0.02)
        )

        # DiT blocks
        cond_dim = global_cond_dim + timestep_embed_dim
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)
        self.action_inds = (0, action_len)
        self.next_obs_inds = (action_len, action_len + obs_len)

        # AdaLN-specific weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Base initialization
        self.apply(init_weights)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.obs_patchifier.patch_encoder.weight.data
        nn.init.normal_(w.view([w.shape[0], -1]), mean=0.0, std=0.02)
        nn.init.constant_(self.obs_patchifier.patch_encoder.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(self, global_cond, action, action_t, next_obs, next_obs_t):
        # Encode inputs
        action_embed = self.action_encoder(action)
        next_obs_embed = self.obs_patchifier(next_obs)

        # Expand and encode timesteps
        if len(action_t.shape) == 0:
            action_t = action_t.expand(action.shape[0]).to(
                dtype=torch.long, device=action.device
            )
        if len(next_obs_t.shape) == 0:
            next_obs_t = next_obs_t.expand(next_obs.shape[0]).to(
                dtype=torch.long, device=next_obs.device
            )
        temb = self.timestep_embedding(action_t, next_obs_t)

        # Forward through model
        registers = self.registers.expand(next_obs.shape[0], -1, -1)
        x = torch.cat((action_embed, next_obs_embed, registers), dim=1)
        x = x + self.pos_embed
        cond = torch.cat((global_cond, temb), dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)

        # Extract action and next observation noise predictions
        action_noise_pred = x[:, self.action_inds[0] : self.action_inds[1]]
        next_obs_noise_pred = x[:, self.next_obs_inds[0] : self.next_obs_inds[1]]

        # Decode outputs
        action_noise_pred = self.action_decoder(action_noise_pred)
        next_obs_noise_pred = self.obs_patchifier.unpatchify(next_obs_noise_pred)
        return action_noise_pred, next_obs_noise_pred


class UnifiedWorldModel(nn.Module):
    def __init__(
        self,
        action_len: int,
        action_dim: int,
        obs_encoder: UWMObservationEncoder,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        latent_patch_shape: tuple[int, ...] = (2, 4, 4),
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        num_registers: int = 8,
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        force_next_obs_t_max: bool = False,
        use_logprob: bool = False,
        logprob_eta: float = 0.3,
        logprob_drop_last: bool = True,
    ):
        """
        Assumes rgb input: (B, T, H, W, C) uint8 image
        Assumes low_dim input: (B, T, D)
        """

        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.action_shape = (action_len, action_dim)

        # Image augmentation
        self.obs_encoder = obs_encoder
        self.latent_img_shape = self.obs_encoder.latent_img_shape()

        # Diffusion noise prediction network
        global_cond_dim = self.obs_encoder.feat_dim()
        image_shape = self.latent_img_shape[2:]
        num_views, num_chans = self.latent_img_shape[:2]
        self.noise_pred_net = DualNoisePredictionNet(
            global_cond_dim=global_cond_dim,
            image_shape=image_shape,
            patch_shape=latent_patch_shape,
            num_chans=num_chans,
            num_views=num_views,
            action_len=action_len,
            action_dim=action_dim,
            embed_dim=embed_dim,
            timestep_embed_dim=timestep_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            num_registers=num_registers,
        )

        # Diffusion scheduler
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.force_next_obs_t_max = force_next_obs_t_max
        self.use_logprob = use_logprob
        self.logprob_eta = logprob_eta
        self.logprob_drop_last = logprob_drop_last
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )

    def forward(
        self,
        obs_dict,
        next_obs_dict,
        action,
        action_mask=None,
        goal_obs_dict=None,
    ):
        batch_size, device = action.shape[0], action.device

        # Encode observations
        obs, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict, goal_obs_dict=goal_obs_dict
        )

        # Sample diffusion timestep for action
        action_noise = torch.randn_like(action)
        action_t = torch.randint(
            low=0, high=self.num_train_steps, size=(batch_size,), device=device
        ).long()
        if action_mask is not None:
            action_t[~action_mask] = self.num_train_steps - 1
        noisy_action = self.noise_scheduler.add_noise(action, action_noise, action_t)

        # Sample diffusion timestep for next observation
        next_obs_noise = torch.randn_like(next_obs)
        if self.force_next_obs_t_max:
            next_obs_t = torch.full(
                (batch_size,),
                self.num_train_steps - 1,
                device=device,
                dtype=torch.long,
            )
        else:
            next_obs_t = torch.randint(
                low=0, high=self.num_train_steps, size=(batch_size,), device=device
            ).long()
        noisy_next_obs = self.noise_scheduler.add_noise(
            next_obs, next_obs_noise, next_obs_t
        )

        # Diffusion loss
        action_noise_pred, next_obs_noise_pred = self.noise_pred_net(
            obs, noisy_action, action_t, noisy_next_obs, next_obs_t
        )
        action_loss = F.mse_loss(action_noise_pred, action_noise)
        dynamics_loss = F.mse_loss(next_obs_noise_pred, next_obs_noise)
        loss = action_loss + dynamics_loss

        # Logging
        info = {
            "loss": loss.item(),
            "action_loss": action_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
        }
        return loss, info

    @torch.no_grad()
    def sample(self, obs_dict, goal_obs_dict=None):
        return self.sample_marginal_action(obs_dict, goal_obs_dict=goal_obs_dict)

    @torch.no_grad()
    def sample_forward_dynamics(self, obs_dict, action, goal_obs_dict=None):
        # Encode observations
        obs = self.obs_encoder.encode_curr_obs(obs_dict, goal_obs_dict=goal_obs_dict)

        # Initialize next observation sample
        next_obs_sample = torch.randn(
            (obs.shape[0],) + self.latent_img_shape, device=obs.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        action_t = self.noise_scheduler.timesteps[-1]
        for next_obs_t in self.noise_scheduler.timesteps:
            _, next_obs_noise_pred = self.noise_pred_net(
                obs, action, action_t, next_obs_sample, next_obs_t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, next_obs_t, next_obs_sample
            ).prev_sample
        return next_obs_sample

    @torch.no_grad()
    def sample_forward_dynamics_with_logprob(
        self,
        obs_dict,
        action,
        goal_obs_dict=None,
        eta: float = None,
        drop_last: bool = None,
    ):
        """
        Sample next observation latent and return per-step DDIM transition log-probability.
        """
        if eta is None:
            eta = self.logprob_eta
        if drop_last is None:
            drop_last = self.logprob_drop_last

        obs = self.obs_encoder.encode_curr_obs(obs_dict, goal_obs_dict=goal_obs_dict)
        next_obs_sample = torch.randn(
            (obs.shape[0],) + self.latent_img_shape, device=obs.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        action_t = timesteps[-1]

        all_latents = [next_obs_sample]
        all_log_probs = []
        for next_obs_t in timesteps:
            _, next_obs_noise_pred = self.noise_pred_net(
                obs, action, action_t, next_obs_sample, next_obs_t
            )
            prev_sample, log_prob = ddim_step_with_logprob(
                scheduler=self.noise_scheduler,
                model_output=next_obs_noise_pred,
                timestep=next_obs_t,
                sample=next_obs_sample,
                eta=eta,
            )
            all_log_probs.append(log_prob)
            all_latents.append(prev_sample)
            next_obs_sample = prev_sample

        latents = torch.stack(all_latents, dim=1)
        next_latents = latents[:, 1:]
        curr_latents = latents[:, :-1]
        log_probs = torch.stack(all_log_probs, dim=1)
        timesteps = timesteps.to(obs.device).unsqueeze(0).expand(obs.shape[0], -1)

        if drop_last:
            curr_latents = curr_latents[:, :-1]
            next_latents = next_latents[:, :-1]
            log_probs = log_probs[:, :-1]
            timesteps = timesteps[:, :-1]

        return {
            "sample": next_obs_sample,
            "timesteps": timesteps,
            "latents": curr_latents,
            "next_latents": next_latents,
            "log_probs": log_probs,
        }

    @torch.no_grad()
    def sample_forward_dynamics_for_rl(self, obs_dict, action, goal_obs_dict=None):
        """
        Config switch:
          - use_logprob=True  -> return dict with logprob trajectory
          - use_logprob=False -> return dict with only final sample
        """
        if self.use_logprob:
            return self.sample_forward_dynamics_with_logprob(
                obs_dict=obs_dict,
                action=action,
                goal_obs_dict=goal_obs_dict,
            )
        return {"sample": self.sample_forward_dynamics(obs_dict, action, goal_obs_dict)}

    def evaluate_forward_dynamics_logprob(
        self,
        obs_dict,
        action,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        timesteps: torch.Tensor,
        goal_obs_dict=None,
        eta: float = None,
    ) -> torch.Tensor:
        """
        Recompute per-step DDIM transition log-probability for an existing trajectory.
        This is the training-time counterpart of sampling-time `*_with_logprob`.

        Args:
            obs_dict: current observation dict
            action: [B, action_len, action_dim]
            latents: [B, S, ...]    (x_t)
            next_latents: [B, S, ...] (x_{t-1})
            timesteps: [B, S] or [S]
        Returns:
            log_probs: [B, S]
        """
        if eta is None:
            eta = self.logprob_eta

        # Ensure scheduler state is initialized even when this model instance
        # has not gone through sampling (e.g. frozen ref model for KL).
        inference_steps = self.num_inference_steps
        if inference_steps is None:
            # Fallback path: infer from provided trajectory length.
            # When sampling used drop_last=True, timesteps excludes the final 0 step,
            # so we add one step back for scheduler reconstruction.
            inferred_from_timesteps = int(timesteps.shape[-1])
            if bool(getattr(self, "logprob_drop_last", False)):
                ts_min = int(torch.as_tensor(timesteps).min().item())
                if ts_min > 0:
                    inferred_from_timesteps += 1
            if timesteps.ndim == 1:
                inference_steps = inferred_from_timesteps
            elif timesteps.ndim == 2:
                inference_steps = inferred_from_timesteps
            else:
                raise ValueError(
                    "`num_inference_steps` is None and failed to infer from "
                    f"`timesteps` shape={tuple(timesteps.shape)}"
                )
        inference_steps = int(inference_steps)
        if inference_steps <= 0:
            raise ValueError(
                f"`num_inference_steps` must be positive, got {inference_steps}"
            )
        self.noise_scheduler.set_timesteps(inference_steps)

        if latents.shape != next_latents.shape:
            raise ValueError(
                f"`latents` and `next_latents` must have same shape, got "
                f"{tuple(latents.shape)} vs {tuple(next_latents.shape)}"
            )
        if latents.ndim < 3:
            raise ValueError(
                f"`latents` must have shape [B,S,...], got {tuple(latents.shape)}"
            )

        batch_size, num_steps = latents.shape[:2]
        if timesteps.ndim == 1:
            if timesteps.shape[0] != num_steps:
                raise ValueError(
                    f"1D timesteps length mismatch: {timesteps.shape[0]} vs S={num_steps}"
                )
            timesteps = timesteps.unsqueeze(0).expand(batch_size, -1)
        elif timesteps.ndim == 2:
            if timesteps.shape[0] != batch_size or timesteps.shape[1] != num_steps:
                raise ValueError(
                    f"2D timesteps shape mismatch: {tuple(timesteps.shape)} vs "
                    f"(B,S)=({batch_size},{num_steps})"
                )
        else:
            raise ValueError(
                f"`timesteps` must be [S] or [B,S], got {tuple(timesteps.shape)}"
            )
        timesteps = timesteps.to(device=latents.device, dtype=torch.long)

        obs = self.obs_encoder.encode_curr_obs(obs_dict, goal_obs_dict=goal_obs_dict)

        # Keep action_t consistent with sampling logic in sample_forward_dynamics_with_logprob:
        # action_t should come from the full scheduler horizon's terminal step, not from
        # possibly drop_last-truncated `timesteps`.
        scheduler_action_t = self.noise_scheduler.timesteps[-1]
        if not torch.is_tensor(scheduler_action_t):
            scheduler_action_t = torch.tensor(
                scheduler_action_t, device=latents.device, dtype=torch.long
            )
        action_t = scheduler_action_t.to(device=latents.device, dtype=torch.long).expand(
            batch_size
        )
        log_probs = []
        for step_idx in range(num_steps):
            step_t = timesteps[:, step_idx]
            step_latent = latents[:, step_idx]
            step_next = next_latents[:, step_idx]
            _, next_obs_noise_pred = self.noise_pred_net(
                obs, action, action_t, step_latent, step_t
            )
            _, step_log_prob = ddim_step_with_logprob(
                scheduler=self.noise_scheduler,
                model_output=next_obs_noise_pred,
                timestep=step_t,
                sample=step_latent,
                eta=eta,
                prev_sample=step_next,
            )
            log_probs.append(step_log_prob)
        return torch.stack(log_probs, dim=1)

    @torch.no_grad()
    def sample_inverse_dynamics(self, obs_dict, next_obs_dict, goal_obs_dict=None):
        # Encode observations
        obs_feat, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict, goal_obs_dict=goal_obs_dict
        )

        # Initialize action sample
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_obs_t = self.noise_scheduler.timesteps[-1]
        for action_t in self.noise_scheduler.timesteps:
            action_noise_pred, _ = self.noise_pred_net(
                obs_feat, action_sample, action_t, next_obs, next_obs_t
            )
            action_sample = self.noise_scheduler.step(
                action_noise_pred, action_t, action_sample
            ).prev_sample
        return action_sample

    @torch.no_grad()
    def sample_marginal_next_obs(self, obs_dict, goal_obs_dict=None):
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict, goal_obs_dict=goal_obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        action_t = self.noise_scheduler.timesteps[0]
        for t in self.noise_scheduler.timesteps:
            _, next_obs_noise_pred = self.noise_pred_net(
                obs_feat, action_sample, action_t, next_obs_sample, t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample
        return next_obs_sample

    @torch.no_grad()
    def sample_marginal_next_obs_with_logprob(
        self,
        obs_dict,
        goal_obs_dict=None,
        eta: float = None,
        drop_last: bool = None,
    ):
        """
        Marginally sample next observation latent and return DDIM transition log-probability.
        """
        if eta is None:
            eta = self.logprob_eta
        if drop_last is None:
            drop_last = self.logprob_drop_last

        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict, goal_obs_dict=goal_obs_dict)
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        action_t = timesteps[0]

        all_latents = [next_obs_sample]
        all_log_probs = []
        for t in timesteps:
            _, next_obs_noise_pred = self.noise_pred_net(
                obs_feat, action_sample, action_t, next_obs_sample, t
            )
            prev_sample, log_prob = ddim_step_with_logprob(
                scheduler=self.noise_scheduler,
                model_output=next_obs_noise_pred,
                timestep=t,
                sample=next_obs_sample,
                eta=eta,
            )
            all_log_probs.append(log_prob)
            all_latents.append(prev_sample)
            next_obs_sample = prev_sample

        latents = torch.stack(all_latents, dim=1)
        next_latents = latents[:, 1:]
        curr_latents = latents[:, :-1]
        log_probs = torch.stack(all_log_probs, dim=1)
        timesteps = timesteps.to(obs_feat.device).unsqueeze(0).expand(obs_feat.shape[0], -1)

        if drop_last:
            curr_latents = curr_latents[:, :-1]
            next_latents = next_latents[:, :-1]
            log_probs = log_probs[:, :-1]
            timesteps = timesteps[:, :-1]

        return {
            "sample": next_obs_sample,
            "timesteps": timesteps,
            "latents": curr_latents,
            "next_latents": next_latents,
            "log_probs": log_probs,
        }

    @torch.no_grad()
    def sample_marginal_next_obs_for_rl(self, obs_dict, goal_obs_dict=None):
        """
        Config switch:
          - use_logprob=True  -> return dict with logprob trajectory
          - use_logprob=False -> return dict with only final sample
        """
        if self.use_logprob:
            return self.sample_marginal_next_obs_with_logprob(
                obs_dict=obs_dict,
                goal_obs_dict=goal_obs_dict,
            )
        return {"sample": self.sample_marginal_next_obs(obs_dict, goal_obs_dict)}

    @torch.no_grad()
    def sample_marginal_action(self, obs_dict, goal_obs_dict=None):
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict, goal_obs_dict=goal_obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_obs_t = self.noise_scheduler.timesteps[0]
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, _ = self.noise_pred_net(
                obs_feat, action_sample, t, next_obs_sample, next_obs_t
            )
            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample
        return action_sample

    @torch.no_grad()
    def sample_joint(self, obs_dict, goal_obs_dict=None):
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict, goal_obs_dict=goal_obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, next_obs_noise_pred = self.noise_pred_net(
                obs_feat, action_sample, t, next_obs_sample, t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample
            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample
        return next_obs_sample, action_sample

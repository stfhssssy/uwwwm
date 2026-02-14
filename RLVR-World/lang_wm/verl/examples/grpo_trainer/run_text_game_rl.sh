# using 8x80G A100 GPUs

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=thuml/bytesized32-world-model-cot/train_state_difference_gold.parquet \
    data.val_files=thuml/bytesized32-world-model-cot/test_state_difference.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=7192 \
    data.max_response_length=4096 \
    data.need_filter=False \
    actor_rollout_ref.model.path=thuml/bytesized32-world-model-base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=3 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=3 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=verl_grpo_text_game_simulator \
    trainer.experiment_name=grpo_text_game_simulator_binary_reward \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=log/rlvr_text_game_simulator_experiment \
    actor_rollout_ref.rollout.max_num_batched_tokens=11288 \
    +data.sample_no_gold_data=True \
    +data.sample_no_gold_data_num=7278 \   # 1000 for task-specific reward
    +data.sample_no_gold_data_file=thuml/bytesized32-world-model-cot/train_state_difference_non_gold.parquet \
    +data.dataset_type=text_game_dataset \
    +reward_model.text_game_reward_type=binary \  # task_specific for task-specific reward
    $@ | tee verl_vgpt.log

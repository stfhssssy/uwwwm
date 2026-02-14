# using 4x80G A100 GPUs

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=thuml/bytesized32-world-model-cot/generated_cot.parquet \
    data.val_files=thuml/bytesized32-world-model-cot/generated_cot.parquet \
    data.train_batch_size=16 \
    data.prompt_key=prompt \
    data.response_key=reward_model \
    +data.prompt_dict_keys=['content'] \
    +data.response_dict_keys=['ground_truth'] \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=11384 \
    model.partial_pretrain=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    trainer.default_local_dir=log/sft_text_game_simulator_experiment \
    trainer.project_name=text_game_simulator_sft \
    trainer.experiment_name=text_game_simulator_deepseek_generated_data_sft \
    trainer.total_epochs=15 \
    trainer.logger=['console','wandb'] \
    model.lora_rank=32 \
    model.lora_alpha=16 $@
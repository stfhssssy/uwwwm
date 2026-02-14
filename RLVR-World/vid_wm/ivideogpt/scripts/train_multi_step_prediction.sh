# using 8x40G A100 GPUs

accelerate launch train_vgpt.py \
    --per_device_train_batch_size 4 \
    --config_name configs/vgpt/ctx_llama_small.json \
    --dataset_path /dev/null \
    --pretrained_model_name_or_path thuml/rt1-compressive-tokenizer \
    --output_dir trm-output \
    --skip_first_val \
    --exp_name vgpt_small_ctx_msp8_head12 \
    --processor_type ctx_msp \
    --segment_length 8 $@
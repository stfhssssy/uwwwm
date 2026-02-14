# using one GPU

python eval_runenv.py --per_device_eval_batch_size 1 \
    --config_name configs/vgpt/ctx_llama_small.json\
    --dataset_path /dev/null  \
    --pretrained_model_name_or_path thuml/rt1-compressive-tokenizer \
    --pretrained_transformer_path thuml/rt1-world-model-multi-step-rlvr \
    --processor_type ctx_msp \
    --max_decode_batchsize 1 \
    --segment_length 8 \
    --gpu_memory_utilization 0.75 \
    --repetition_penalty 1.2 \
    --output_dir policy_eval \
    --policy_model_path pretrained_models/rt_1_tf_trained_for_000400120 \
    --task_instruction "open middle drawer" $@
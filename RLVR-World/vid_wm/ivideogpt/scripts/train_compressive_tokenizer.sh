# using 16x40G A100 GPUs

accelerate launch train_ctx_tokenizer.py \
    --exp_name ctx_cnn_fsq12_frac_res320_seg8 \
    --dataset_path /dev/null \
    --train_batch_size 1 --gradient_accumulation_steps 1 --log_code_util \
    --resolution 256 320 \
    --output_dir vqgan-output \
    --vae_loss l1 --disc_weight 0.1 --perc_weight 1.0 \
    --start_global_step 0 --disc_start 10000 --max_train_steps 600000 \
    --discr_learning_rate 5e-4 --learning_rate 5e-4 \
    --disc_depth 6 --segment_length 8 $@
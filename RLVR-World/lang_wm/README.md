# Language World Models with RLVR

## Method Overview

![language world model](assets/lang_wm.png)

## Installation

```bash
# Python 3.10 or 3.11 recommended
conda create -n lang_wm python=3.10
cd verl
pip install -e .
```

## Data Preparation

### Text Game

All data has been uploaded to [Hugging Face](https://huggingface.co/datasets/thuml/bytesized32-world-model-cot).

You can either download them directly or generate them yourself by following instructions in [``data_process/text_game``](data_process/text_game).

###  Web Page

All data has been uploaded to [Hugging Face](https://huggingface.co/datasets/thuml/webarena-world-model-cot).

## Supervised Fine-Tuning (SFT)

### Text Game

```bash
bash verl/examples/sft/text_game/run_text_game_sft.sh
```

### Web Page

```bash
bash verl/examples/sft/web_agent/run_web_agent_sft.sh
```

After training, the model will be saved in `default_local_dir` specified in the script.

To merge the LoRA weights into the base model, run the following command:

```bash
python verl/merge_lora.py
```

You have to specify the directory for LoRA weights in the script.

## Post-training with RLVR

### Text Game

Run the following command.

```bash
bash examples/grpo_trainer/run_text_game_rl.sh \
    +data.sample_no_gold_data_num=7278 \   # 1000 for task-specific reward
    +reward_model.text_game_reward_type=binary  # =task_specific for task-specific reward
```

This command uses binary reward by default. If you want to use the task-specific reward described in the paper, simply modify the two parameters ``data.sample_no_gold_data_num`` and ``reward_model.text_game_reward_type`` as indicated in the comments.

### Web Page

```bash
bash examples/grpo_trainer/run_web_agnet_rl.sh
```

The trained model will be saved in `default_local_dir` specified in the script.

To merge weights, run the following command:

```bash
python verl/scripts/model_merger.py --local_dir log/xxxx/checkpoints/global_step_xxxx/actor --output_dir <output_dir> --backend fsdp --hf_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
```

## Downstream Task: Model Predictive Control for Web Agents

Code and instructions can be found in [``webagent``](webagent).

## Acknowledgements

Our `verl` codebase is forked from commit [`fbad52`](https://github.com/volcengine/verl/tree/fbad52e1204c84f277b4e94f2f236b51b0ebaff4) of official repo.

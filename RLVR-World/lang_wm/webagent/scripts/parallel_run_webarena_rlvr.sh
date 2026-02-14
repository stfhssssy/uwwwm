#!/bin/bash

export DATASET=webarena

# change <your-openai-key> to the real OpenAI key
export OPENAI_API_KEY="<your-openai-key>"
export OPENAI_API_BASE="https://api.deepseek.com/v1"

# change <your-server-hostname> to the real host name of your AWS machine
export SHOPPING="http://<your-server-hostname>:7770"
export SHOPPING_ADMIN="http://<your-server-hostname>:7780/admin"
export REDDIT="http://<your-server-hostname>:9999"
export GITLAB="http://<your-server-hostname>:8023"
export MAP="http://<your-server-hostname>:3000"
export WIKIPEDIA="http://<your-server-hostname>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://<your-server-hostname>:4399"

# change <your-world_model_name> to the real world model name.
# You can get it by running `curl http://<your-server-ip>:8000/v1/models`
# To do this, you should start the vllm service first by running
# python -m vllm.entrypoints.openai.api_server
# --model <path-to-your-world-model>
# -port 8000
world_model_name="<your-world_model_name>"
world_model_url="http://<your-server-ip>:8000/v1"
value_model_name=""
value_model_url=""

model="deepseek-chat"
value_function="deepseek-chat"

max_depth=2  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=3
vf_budget=20
agent="world_model"
world_model_training=True
value_model_training=False
my_world_model=True
next_state_format="description_with_tao"
result_dir="log"
instruction_path="agent/prompts/jsons/p_cot_id_actree_2s_no_na.json"
state_prediction_prompt_path="agent/prompts/jsons/state_prediction/rlvr_worldmy_world_model_prompt.json"
value_function_prompt_path="agent/prompts/jsons/value_function/text_only_value_function_likert.json"

mkdir "${result_dir}"
mkdir "${result_dir}/logs"

### Code to run the experiments
function run_job() {
    local start_idx=$1
    local end_idx=$2
    local job_num=$3

    if [ -f logs/wma_${next_state_format}_format_job_${job_num}.log ]; then
        echo "----------------------------------------" >> logs/wma_${next_state_format}_format_job_${job_num}.log
        echo "New log entry started at $(date)" >> logs/wma_${next_state_format}_format_job_${job_num}.log
        echo "----------------------------------------" >> logs/wma_${next_state_format}_format_job_${job_num}.log
    else
        touch ${result_dir}/logs/wma_${next_state_format}_format_job_${job_num}.log
    fi
    nohup python run_w_world_model.py \
    --instruction_path $instruction_path \
    --test_start_idx $start_idx \
    --test_end_idx $end_idx \
    --model $model \
    --agent_type $agent   --max_depth $max_depth  --branching_factor $branching_factor  --vf_budget $vf_budget   \
    --result_dir $result_dir \
    --test_config_base_dir=config_files/wa/test_webarena \
    --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
    --top_p 0.95   --temperature 1.0  --max_steps $max_steps --value_function $value_function\
    --state_prediction_prompt_path $state_prediction_prompt_path --value_function_prompt_path $value_function_prompt_path --total_indices $total_indices\
    --next_state_format $next_state_format \
    $( [ "$world_model_training" = True ] && echo "--world_model_training" ) \
    $( [ "$world_model_training" = True ] && echo "--world_model_name $world_model_name" ) \
    $( [ "$world_model_training" = True ] && echo "--world_model_url $world_model_url" ) \
    $( [ "$value_model_training" = True ] && echo "--value_model_training" ) \
    $( [ "$value_model_training" = True ] && echo "--value_model_name $value_model_name" ) \
    $( [ "$value_model_training" = True ] && echo "--value_model_url $value_model_url" ) \
    >> ${result_dir}/logs/wma_${next_state_format}_format_job_${job_num}.log 2>&1 &
}

total_indices=812
indices_per_thread=1
batch_size=40
batch_end=0
current_start=0
job_count=0

echo "start evaluation..."

while [ "$current_start" -lt "$total_indices" ]; do
  batch_end=$((batch_end + batch_size))
  if [ "$batch_end" -gt "$total_indices" ]; then
      batch_end=$total_indices
  fi
  echo "a new round has been started...\n"
  while [ "$current_start" -lt "$batch_end" ]; do
      current_end=$((current_start + indices_per_thread))
      if [ "$current_end" -gt "$total_indices" ]; then
          current_end=$total_indices
      fi

      # Run the job
      run_job $current_start $current_end $job_count

      ((job_count++))

      # Increment start index for next job
      current_start=$current_end
  done
  ### Wait for all jobs to complete
  wait
done
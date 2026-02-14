# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

import pandas as pd
import argparse
import json
from tqdm import tqdm
import os
from experiments.process_jsonl_train import get_state_diff_detail_v2

if not os.path.exists("./deepseek_generate_response"):
    os.mkdir("./deepseek_generate_response")

dataframe = pd.read_parquet("dev/null/train_state_difference_all_data.parquet")  # TODO：the path to the original data
prompts = dataframe["prompt"].tolist()
responses = dataframe["reward_model"].tolist()
extra_infos = dataframe["extra_info"].tolist()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=-1)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()
    return args


args = parse_args()
start_index = args.start_index
end_index = args.end_index
assert start_index >= 0
gold_num = 0

client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")

for i in tqdm(range(start_index, min(end_index, len(prompts)))):
    current_index_str = str(i).zfill(6)
    if os.path.exists(f"./deepseek_generate_response/{current_index_str}.json"):
        continue
    prompt_content = prompts[i][0]["content"]

    # add gold state info
    diffs_gold = get_state_diff_detail_v2(json.loads(extra_infos[i]["data_state"]), json.loads(responses[i]["ground_truth"]))
    # process gold state
    gold_record = []
    for key, state_1, state_2, state_code in diffs_gold["modified"]:
        try:
            uuid = state_1["uuid"]
        except:
            uuid = state_2["uuid"]

        if state_code != 1:
            gold_record.append((uuid, key))
    # if len(gold_record) == 0:
    #     continue

    gold_num += 1

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"{prompt_content}\n"},
        ],
        stream=False
    )
    with open(f"./deepseek_generate_response/{current_index_str}.json", "w") as file:
        json_data = {
            "index": i,
            "prompt": prompt_content,
            "response": response.choices[0].message.content,
            "reasoning_response": response.choices[0].message.reasoning_content
        }
        json.dump(json_data, file, indent=4)

print(gold_num)

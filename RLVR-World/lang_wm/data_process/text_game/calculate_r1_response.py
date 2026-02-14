import os
import pandas as pd
import json
from text_game import compute_score
from tqdm import tqdm
import random

files = os.listdir("./deepseek_generate_response")

dataframe = pd.read_parquet("dev/null/train_state_difference_all_data.parquet")  # TODO：the path to the original data
prompts = dataframe["prompt"].tolist()
responses = dataframe["reward_model"].tolist()
extra_infos = dataframe["extra_info"].tolist()

gold_score_record = []
gold_num = 0
full_score_record = []

selected_files = []

for file in tqdm(files):
    one_json = os.path.join("./deepseek_generate_response", file)
    try:
        with open(one_json, "r") as f:
            one_answer = json.load(f)
    except:
        print(f"invalid file: {file}")
        continue

    index = int(one_answer["index"])
    prediction = one_answer["response"]
    # reasoning_len = one_answer["reasoning_response"]
    ground_truth = responses[index]["ground_truth"]
    extra_info = extra_infos[index]

    one_result = compute_score(prediction, ground_truth, extra_info)

    if isinstance(one_result, list):
        full_score = one_result[1]
        full_score_record.append(full_score)
        if len(one_result) == 3:
            gold_score = one_result[0]
            gold_score_record.append(gold_score)
            if gold_score == 1:
                gold_num += 1
                selected_files.append([index, file, "gold"])
        elif full_score == 1:
            if random.randint(0, 100) == 3:
                selected_files.append([index, file, "correct"])
    else:
        continue

print(f"gold_score_record: len:{len(gold_score_record)}, sum:{sum(gold_score_record)}, average:{sum(gold_score_record)/len(gold_score_record)}")
print(f"full_score_record: len:{len(full_score_record)}, sum:{sum(full_score_record)}, average:{sum(full_score_record)/len(full_score_record)}")
print(f"gold_num: {gold_num}")

print(len(selected_files))

with open(f'./r1_good_response_record.json', "w") as ff:
    json.dump({"selected:": selected_files}, ff, indent=4)


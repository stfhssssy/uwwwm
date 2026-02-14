import json
import pandas as pd
import os
from datasets import Dataset
from tqdm import tqdm
import warnings


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        kwargs['eos_token'] = '<end_of_turn>'
        kwargs['eos_token_id'] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


tokenizer = hf_tokenizer("/dev/null/deepseek-r1-distill-qwen-1-5B")  # TODO：the path to the qwen model

dataframe = pd.read_parquet("/dev/null/train_state_difference_all_data.parquet")  # TODO：the path to the original data
prompts = dataframe["prompt"].tolist()
responses = dataframe["reward_model"].tolist()
extra_infos = dataframe["extra_info"].tolist()

sft_data = []

with open(f'./r1_good_response_record.json', "r") as ff:
    good_datas = json.load(ff)

filtered_long_data_num = 0

record_list = good_datas["selected:"]
for one_record in tqdm(record_list):
    index = int(one_record[0])
    file = one_record[1]

    one_json = os.path.join("./deepseek_generate_response", file)
    with open(one_json, "r") as f:
        one_answer = json.load(f)

    prediction = one_answer["response"]
    reasoning_response = one_answer["reasoning_response"]
    prompt = one_answer["prompt"]
    extra_info = extra_infos[index]

    if len(tokenizer(reasoning_response + prediction + tokenizer.eos_token, return_tensors='pt', add_special_tokens=False)['input_ids'][0]) > 4000:
        filtered_long_data_num += 1
        continue

    one_data = {
        "data_source": "llm_simulator",
        "prompt": [{
            "role": "user",
            "content": prompt
        }],
        "ability": "next state prediction",  # fixme
        "reward_model": {
            "style": "rule",
            "ground_truth": reasoning_response + prediction
        },
        "extra_info": extra_info,
    }

    sft_data.append(one_data)

print(filtered_long_data_num)
df = pd.DataFrame(sft_data)
sft_dataset = Dataset.from_pandas(df)
sft_dataset.to_parquet(os.path.join('/dev/null/', 'deepseek_sft_data.parquet'))  # TODO: where to save the path

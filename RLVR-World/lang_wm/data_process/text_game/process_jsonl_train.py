import tiktoken             # pip install tiktoken
import os
import random
import json
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from math import ceil
import sys

sys.path.insert(0, "dev/null/GPT_simulator")

from scripts.evaluate import evaluate, make_game_state, make_game_state_partial, evaluate_score


# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Get the number of tokens for a string, measured using tiktoken
def getTokenLength(strIn):
    tokens = tokenizer.encode(strIn)
    numTokens = len(tokens)
    return numTokens


def compare_dict(dict_1, dict_2):
    """ compare if two dictionary are the same """
    if len(dict_1) != len(dict_2):
        return False
    for key in dict_1:
        if key not in dict_2:
            return False
        # We don't care it is a list/tuple
        if type(dict_1[key]) in [list, tuple] and type(dict_2[key]) in [list, tuple]:
            if list(dict_1[key]) != list(dict_2[key]):
                return False
        elif type(dict_1[key]) == dict and type(dict_2[key]) == dict:
            if not compare_dict(dict_1[key], dict_2[key]):
                return False
        elif dict_1[key] != dict_2[key]:
            return False
    return True


def make_state_for_comprison(states):
    obj_dict = {}
    score = {}
    for state in states:
        if 'uuid' in state:
            obj_dict[state['uuid']] = state
        elif 'score' in state:
            score = state
    return obj_dict, score


def get_state_diff_detail_v2(state_1, state_2):
    """ list the diffs in each object property

    This version is used for the experiments that split action and tick apart
    """
    state_1, _ = make_state_for_comprison(state_1["game_state"])

    state_2, _ = make_state_for_comprison(state_2["game_state"])

    diffs = {"added":[], "removed":[], "modified":[], "same":[]}

    # compare objects
    for uuid in state_1:
        is_same = True
        if uuid not in state_2:
            diffs["removed"].append((state_1[uuid], None)) # an object is removed
            is_same = False
            continue
        # compare properties
        property_1 = state_1[uuid]["properties"]
        property_2 = state_2[uuid]["properties"]

        for key in property_1:
            if key not in property_2:
                diffs["modified"].append((key, state_1[uuid], None, 2))
                is_same = False
            else:
                if type(property_1[key]) in [list, tuple] and type(property_2[key]) in [list, tuple]:
                    # We don't care it is a list/tuple
                    if list(property_1[key]) != list(property_2[key]):
                        diffs["modified"].append((key, state_1[uuid], state_2[uuid], 0))
                        is_same = False
                    else:
                        diffs["modified"].append((key, state_1[uuid], state_2[uuid], 1))
                elif type(property_1[key]) == dict and type(property_2[key]) == dict:
                    if not compare_dict(property_1[key], property_2[key]):
                        diffs["modified"].append((key, state_1[uuid], state_2[uuid], 0))
                        is_same = False
                    else:
                        diffs["modified"].append((key, state_1[uuid], state_2[uuid], 1))
                else:
                    if property_1[key] != property_2[key]:
                        diffs["modified"].append((key, state_1[uuid], state_2[uuid], 0))
                        is_same = False
                    else:
                        diffs["modified"].append((key, state_1[uuid], state_2[uuid], 1))

        for key in property_2:
            if key not in property_1:
                diffs["modified"].append((key, state_1[uuid], state_2[uuid], 3))
                is_same = False

        # compare contained objects
        # We don't care the order of contains
        if sorted(state_1[uuid].get("contains", [])) != sorted(state_2[uuid].get("contains", [])):
            diffs["modified"].append(('contains', state_1[uuid], state_2[uuid], 0))
            is_same = False
        else:
            diffs["modified"].append(('contains', state_1[uuid], state_2[uuid], 1))

        if is_same:
            diffs["same"].append(uuid)

    # find new objects created
    for uuid in state_2:
        if uuid not in state_1:
            diffs["added"].append((None, state_2[uuid]))

    return diffs


# Load a python program from a file into a string, and count its tokens using tiktoken
def loadProgram(filename):
    programStr = ""
    with open(filename, 'r') as f:
        programStr = f.read()

        lines = programStr.splitlines()
        program = ""
        for line in lines:
            program += line + "\n"


    tokens = tokenizer.encode(programStr)
    numTokens = len(tokens)

    return program, numTokens


# Postprocessing model response, keep only the code chunck ```python ```
def postProcess(raw_response):
    if raw_response.startswith("```json"):
        return raw_response[8:-4]
    else:
        return raw_response



def recover_game_state_from_partial(curr_state, partial_change, has_score=False):
    recovered_state = {"game_state":[]}
    modified_uuids = {state['uuid']:state for state in partial_change["modified"]}
    if has_score:
        obj_states = curr_state["game_state"][:-1]
    else:
        obj_states = curr_state["game_state"]
    for state in obj_states:
        if state['uuid'] in partial_change["removed"]:
            continue
        if state['uuid'] in modified_uuids:
            recovered_state["game_state"].append(modified_uuids[state['uuid']])
        else:
            recovered_state["game_state"].append(state)

    if has_score:
        if len(partial_change['score']) > 0:
            recovered_state["game_state"].append(partial_change['score'])
        else:
            recovered_state["game_state"].append(curr_state["game_state"][-1])

    return recovered_state

def preprocess_obj_desc(desc):
    per_obj_descs = desc.split('==========')[:-1]
    return "".join(per_obj_descs)

#
#   Main
#

def main(data_filter=None):
    # data_type: ["full", "action", "tick"]
    assert data_filter in ["all", "gold", "no_gold"]
    data_type = "full"
    partial = True
    no_rule = False

    statistics = {}
    filtered_out_num = 0
    saved_gold_num = 0
    saved_no_gold_num = 0

    random.seed(1234)

    with open("experiments/games.json") as f:
        games_all = json.load(f)
        games = games_all['games']
        example_game = games_all['example']
        data_games = games[ceil(len(games)/1)*0:ceil(len(games)/1)*(0+1)]

    print("Number of games: " + str(len(data_games)))
    print("Data_games: " + str(data_games))


    # load dynamic and static data distribution json
    with open("data/dynamic_static_states_per_action.json") as f:
        data_distribution = json.load(f)

    # load state change information (the ids of states that are dynamic)
    with open('data/dynamic_states.json') as f:
        state_change_info = json.load(f)

    # load object rules
    with open(os.path.join("rules/human_written_rules", f"object_rules.json")) as f:
        obj_rules = json.load(f)

    # load action rules
    if data_type != "tick":
        with open(os.path.join("rules/human_written_rules", f"action_rules.json")) as f:
            action_rules = json.load(f)

    # load score rules
    if data_type == "score" or data_type == "full":
        with open(os.path.join("rules/human_written_rules", f'score_rules.json')) as f:
            score_rules = json.load(f)

    if data_type == "action":
        curr_state_key = "current_state"
        next_state_key = "action_state"
    elif data_type == "tick":
        curr_state_key = "action_state"
        next_state_key = "tick_state"
    elif data_type == "score":
        curr_state_key = "current_state"
        next_state_key = "tick_state"
    elif data_type == "full":
        curr_state_key = "current_state"
        next_state_key = "tick_state"

    # open example json
    with open("data/examples.json") as f:
        example_lut = json.load(f)
    if data_type == "score":
        # Load an example that score is changed
        example_score_change = example_lut["score"]
        example_state = make_game_state(example_score_change[curr_state_key])
        example_target_state = make_game_state(example_score_change[next_state_key])
        example_curr_score_state = example_score_change["current_score_state"]
        example_next_score_state = example_score_change["next_score_state"]
        example_task_desc = example_score_change[curr_state_key]["taskDesc"]
        example_action = example_score_change[next_state_key]["lastAction"]

    if data_type == "tick" and len(state_change_info[example_game]['time_change']) > 0:
        # Load one example that the game state is changed over time
        # Make sure the example game has tick change states
        example_time_change = example_lut["tick"]
        example_state = make_game_state(example_time_change[curr_state_key])
        example_target_state = make_game_state(example_time_change[next_state_key])
        if partial:
            example_target_state_partial = make_game_state_partial(example_state, example_target_state)

        example_task_desc = example_time_change["current_state"]["taskDesc"]
        example_UUID_base = example_time_change[curr_state_key]["max_UUID"]

    elif data_type == "action":
        # Load one example that the game state is changed by an action
        example_action_change = example_lut["action"]
        example_state = make_game_state(example_action_change[curr_state_key])
        example_target_state = make_game_state(example_action_change[next_state_key])
        if partial:
            example_target_state_partial = make_game_state_partial(example_state, example_target_state)

        example_action = example_action_change[next_state_key]["lastAction"]
        example_task_desc = example_action_change["current_state"]["taskDesc"]
        example_UUID_base = example_action_change[curr_state_key]["max_UUID"]

    elif data_type == "full":
        # Load one example that the game state is changed by an action
        example_action_change = example_lut["full"]["action"]
        current_score_state = example_action_change["current_score_state"]
        next_score_state = example_action_change["next_score_state"]
        example_state_a = make_game_state(example_action_change[curr_state_key])
        example_state_a["game_state"].append(current_score_state)
        example_target_state_a = make_game_state(example_action_change[next_state_key])
        example_target_state_a["game_state"].append(next_score_state)
        if partial:
            example_target_state_a_partial = make_game_state_partial(example_state_a, example_target_state_a)

        example_action_a = example_action_change[next_state_key]["lastAction"]
        example_task_desc = example_action_change["current_state"]["taskDesc"]
        example_UUID_base_a = example_action_change[curr_state_key]["max_UUID"]

        # Load one example that the game state is changed over time
        # MAKE SURE THE EXAMPLE HAS A TIME CHANGE STATE
        time_change_states = [s for s in state_change_info[example_game]['time_change'] if s not in state_change_info[example_game]['action_change']]
        if len(time_change_states) > 0:
            example_time_change = example_lut["full"]["tick"]

            current_score_state = example_time_change["current_score_state"]
            next_score_state = example_time_change["next_score_state"]
            example_state_t = make_game_state(example_time_change[curr_state_key])
            example_state_t["game_state"].append(current_score_state)
            example_target_state_t = make_game_state(example_time_change[next_state_key])
            example_target_state_t["game_state"].append(next_score_state)
            if partial:
                example_target_state_t_partial = make_game_state_partial(example_state_t, example_target_state_t)

            example_action_t = example_time_change[next_state_key]["lastAction"]
            example_UUID_base_t = example_time_change[curr_state_key]["max_UUID"]
        else:
            example_time_change = None


    example_obj_desc = preprocess_obj_desc(obj_rules[example_game])

    if data_type != "tick":
        example_action_desc= action_rules[example_game]

    if data_type == "score" or data_type == "full":
        example_score_desc = score_rules[example_game]


    with open(os.path.join("data", "train.jsonl")) as f:
        train_data = f.readlines()

    processed_train_data = []
    for i, data_str in tqdm(enumerate(train_data)):
        data = json.loads(data_str)

        game = data["game"]
        if game not in data_games:
            continue

        state_id = data["state_id"]
        if game not in statistics:
            statistics[game] = {"total_errors": 0, "total_states": 0}

        statistics[game]["total_states"] += 1

        total_tokens_prompt = 0

        # load game state data
        data_state = make_game_state(data[curr_state_key])
        if data_type == "full":
            data_state["game_state"].append(data["current_score_state"])
        if data_type == "score":
            data_target = data["next_score_state"]
        elif data_type == "full":
            data_target = make_game_state(data[next_state_key])
            score_target = data["next_score_state"]
            data_target["game_state"].append(score_target)
        else:
            data_target = make_game_state(data[next_state_key])
        data_action = data[next_state_key]["lastAction"]
        data_task_desc = data[curr_state_key]["taskDesc"]
        data_UUID_base = data[curr_state_key]["max_UUID"]
        if data_type == "score":
            data_curr_score = data["current_score_state"]
            data_target_state = make_game_state(data[next_state_key])

        # load rules
        data_obj_desc = preprocess_obj_desc(obj_rules[game])
        if data_type != "tick":
            data_action_desc = action_rules[game]
        if data_type == "score" or data_type == "full":
            data_score_desc = score_rules[game]

        output_str = ''

        if data_type == "score":
            prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to predict the current game score, whether the game is over, and whether the agent wins the game.\n"
        elif data_type == "action":
            prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to decide the new game state after taking an action.\n"
        elif data_type == "tick":
            prompt = "You are a simulator of a text game. Read the task description. Given the current game state in JSON, you need to decide how the game state changes in the next time step (without considering the agent actions). Rules for such changes are described as the tick function of each object.\n"
        elif data_type == "full":
            prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to decide the new game state after taking an action including the game score.\n"

        if data_type != "score":
            prompt += "You may need to create new objects when you predict the new game state. You should assign the uuid of new objects starting from the UUID base given in the instructions."

        if partial and data_type in ("action", "tick"):
            prompt += "Your response should be in the JSON format. It should have two keys: 'modified' and 'removed'. The 'modified' key stores a list of all the object states that are added or changed after taking the action. Keep it an empty list if no object is added or modified. The 'removed' key stores a list of uuids of the objects that are removed. Keep it an empty list if no object is removed.\n"
        elif partial and data_type in ("full"):
            prompt += "Your response should be in the JSON format. It should have three keys: 'modified', 'removed', and 'score'. The 'modified' key stores a list of all the object states that are added or changed after taking the action. Keep it an empty list if no object is added or modified. The 'removed' key stores a list of uuids of the objects that are removed. Keep it an empty list if no object is removed. The 'score' key stores a JSON with three keys: 'score', 'gameOver', and 'gameWon'. 'score' stores the current game score, 'gameOver' stores a bool value on whether the game is over, and 'gameWon' stores a bool value on whether the game is won. \n"
        elif data_type == "score":
            prompt += "Your response should be a JSON with three keys: 'score', 'gameOver', and 'gameWon'. 'score' stores the current game score, 'gameOver' stores a bool value on whether the game is over, and 'gameWon' stores a bool value on whether the game is won.\n"
        else:
            prompt += "Your response should be in the same JSON format as the given game state.\n"

        if not data_type == "full":
            prompt += "Here is an example:\n"

            prompt += "Example game task description:\n"
            prompt += f"{example_task_desc}\n"
            if not no_rule:
                prompt += "Here are the descriptions of all game objects properties in the example game:\n"
                prompt += example_obj_desc.strip()
                prompt += "\n"
                if data_type == "action":
                    prompt += "Here are the descriptions of all game actions in the example game:\n"
                    prompt += example_action_desc.strip()
                    prompt += '\n'

                if data_type == "score":
                    prompt += "Here is a description of the game score function:\n"
                    prompt += example_score_desc.strip()
                    prompt += '\n'

            if data_type == "score":
                prompt += "Here is the previous game state:\n"
            else:
                prompt += "Here is the game state:\n"
            prompt += f'{example_state}\n'
            prompt += '\n'

            if data_type == "score":
                prompt += f"The game score of the preivous state is:\n{example_curr_score_state}\n"
            else:
                prompt += f"The current game UUID base is {example_UUID_base}\n"

            if data_type == "action" or data_type == "score":
                prompt += f"The action to take is: {example_action}\n"


            if data_type == "score":
                prompt += f"Here is the current game state after taking the action:"
                prompt += f"{example_target_state}\n"

            prompt += "The expected response is:\n"
            if data_type == "score":
                prompt += f'{example_next_score_state}\n'
            else:
                if partial:
                    prompt += f'{example_target_state_partial}\n'
                else:
                    prompt += f'{example_target_state}\n'
            prompt += '\n'

        else:

            prompt += "Note that while game states can be changed by actions, some game states may change over the time, which is described in the tick function of each object class. \n"
            if example_time_change is not None:
                prompt += "Here are two examples of both cases. Both examples are from the same example game.\n"

            prompt += "Example game task description:\n"
            prompt += f"{example_task_desc}\n"
            if not no_rule:
                prompt += "Here are the descriptions of all game objects properties in the example game:\n"
                prompt += example_obj_desc.strip()
                prompt += "\n"
                prompt += "Here are the descriptions of all game actions in the example game:\n"
                prompt += example_action_desc.strip()
                prompt += '\n'
                prompt += "Here is a description of the score function of the example game:\n"
                prompt += example_score_desc.strip()
                prompt += '\n'

            if example_time_change is not None:
                # Example 1: a game state is changed by an action
                prompt += "In the first example, the game state is changed by an action:\n"

            prompt += "Here is the game state:\n"
            prompt += f'{example_state_a}\n'
            prompt += '\n'

            prompt += f"The current game UUID base is {example_UUID_base_a}\n"

            prompt += f"The action to take is: {example_action_a}\n"
            prompt += "The expected response is:\n"
            if partial:
                prompt += f'{example_target_state_a_partial}\n'
            else:
                prompt += f'{example_target_state_a}\n'
            prompt += '\n'

            # Example 2: a game state is changed over time
            if example_time_change is not None:
                prompt += "In the second example from the same example game, the game state is changed over the time. Note that while in this example the game state is changed by time only, it is possible that a game state is changed by both an action and time.\n"

                prompt += "Here is the game state:\n"
                prompt += f'{example_state_t}\n'
                prompt += '\n'

                prompt += f"The current game UUID base is {example_UUID_base_t}\n"
                prompt += f"The action to take is: {example_action_t}\n"
                prompt += "The expected response is:\n"
                if partial:
                    prompt += f'{example_target_state_t_partial}\n'
                else:
                    prompt += f'{example_target_state_t}\n'
                prompt += '\n'

        # Task
        prompt += "Here is the game that you need to simulate:\n"
        prompt += "Task Description:\n"
        prompt += f"{data_task_desc}\n"
        if not no_rule:
            prompt += "Here are the descriptions of all game objects properties:\n"
            prompt += data_obj_desc.strip()
            prompt += "\n"
            if data_type == "action" or data_type == "full":
                prompt += "Here are the descriptions of all game actions:\n"
                prompt += data_action_desc.strip()
                prompt += '\n'
            if data_type == "score" or data_type == "full":
                prompt += "Here is a description of the game score function:\n"
                prompt += data_score_desc.strip()
                prompt += '\n'


        if data_type == "score":
            prompt += "Here is the previous game state:\n"
        else:
            prompt += "Here is the game state:\n"
        prompt += f'{data_state}\n'
        prompt += '\n'

        if data_type == "score":
            prompt += f"The game score of the preivous state is:\n{data_curr_score}\n"
        else:
            prompt += f"The current game UUID base is {data_UUID_base}\n"

        if data_type == "action" or data_type == "score" or data_type == "full":
            prompt += f"The action to take is:\n{data_action}\n"

        if data_type == "score":
            prompt += f"Here is the current game state after taking the action:\n"
            prompt += f"{data_target_state}\n"

        output_str += 'Prompt:\n'
        output_str += prompt
        output_str += '\n===================================================\n'

        # print(prompt)
        numTokens_prompt = getTokenLength(prompt)
        total_tokens_prompt += numTokens_prompt
        # print(total_tokens_prompt)

        # add gold state info
        diffs_gold = get_state_diff_detail_v2(data_state, data_target)

        # process gold state
        gold_record = []
        for key, state_1, state_2, state_code in diffs_gold["modified"]:
            try:
                uuid = state_1["uuid"]
            except:
                uuid = state_2["uuid"]

            if state_code != 1:
                gold_record.append((uuid, key))

        if len(gold_record) == 0:
            if data_filter == "gold":
                filtered_out_num += 1
                continue
            saved_no_gold_num += 1
        else:
            if data_filter == "no_gold":
                filtered_out_num += 1
                continue
            saved_gold_num += 1

        one_data = {
            "data_source": "llm_simulator",
            "prompt": [{
                "role": "user",
                "content": prompt + "\nPlease make sure to clearly indicate your answer by starting with 'The final answer is:'.\n",
            }],
            "ability": "next state prediction",  # fixme
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(data_target)
            },
            "extra_info": {
                'split': "train",
                'index': i,
                "data_state": json.dumps(data_state),
                "data_action": data_action,
            },
        }

        processed_train_data.append(one_data)

    df = pd.DataFrame(processed_train_data)
    train_dataset = Dataset.from_pandas(df)
    train_dataset.to_parquet(os.path.join('dev/null/llm-wm/processed_data_for_rl', f'train_state_difference_{data_filter}_data.parquet'))

    print(filtered_out_num)
    print(saved_no_gold_num)
    print(saved_gold_num)



    #
    # results saving
    #


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "gold":
            print("only gold data")
            main(data_filter="gold")
        elif arg == "no_gold":
            print("only no-gold data")
            main(data_filter="no_gold")
        else:
            assert False, "we only support gold or no_gold now, you can change the code yourself if you do need other data"
    else:
        print("whole train data")
        main(data_filter="all")

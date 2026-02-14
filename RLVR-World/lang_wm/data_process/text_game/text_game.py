import json
import dirtyjson
from json_repair import repair_json
import re

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
        #if sorted(state_1[uuid]["contains"]) != sorted(state_2[uuid]["contains"]):
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


def recover_game_state_from_partial(curr_state, partial_change, has_score=False):
    recovered_state = {"game_state":[]}
    if isinstance(partial_change, dict) and "modified" in partial_change:
        modified_uuids = {}
        try:
            for state in partial_change["modified"]:
                try:
                    modified_uuids[state['uuid']] = state
                except:
                    pass
        except:
            modified_uuids = []
    else:
        modified_uuids = []
    if has_score:
        obj_states = curr_state["game_state"][:-1]
    else:
        obj_states = curr_state["game_state"]
    for state in obj_states:
        if isinstance(partial_change, dict) and "removed" in partial_change:
            try:
                if state['uuid'] in partial_change["removed"]:
                    continue
                elif state['uuid'] == partial_change["removed"]:
                    continue
            except:
                pass
        if state['uuid'] in modified_uuids:
            recovered_state["game_state"].append(modified_uuids[state['uuid']])
        else:
            recovered_state["game_state"].append(state)

    if has_score:
        if isinstance(partial_change, dict) and "score" in partial_change:
            try:
                if len(partial_change['score']) > 0:
                    recovered_state["game_state"].append(partial_change['score'])
                else:
                    recovered_state["game_state"].append(curr_state["game_state"][-1])
            except:
                recovered_state["game_state"].append(curr_state["game_state"][-1])

    return recovered_state


def make_state_for_comprison(states):
    obj_dict = {}
    score = {}
    for state in states:
        if 'uuid' in state:
            obj_dict[state['uuid']] = state
        elif 'score' in state:
            score = state
    return obj_dict, score


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


def evaluate(prediction, target, last_action, evaluate_score=False):
    num_errors = 0
    num_errors_score = 0

    num_correct = 0
    num_correct_score = 0

    out_str = ''
    try:
        target, score_t = make_state_for_comprison(target["game_state"])

        prediction, score_p = make_state_for_comprison(prediction["game_state"])

        out_str += f"last_action: {last_action}\n\n"

        # compare objects
        for uuid in target:
            if uuid not in prediction:
                out_str += f"Missing object: {target[uuid]['name']}\n\n"
                num_errors += 1
            else:
                if "properties" not in prediction[uuid]:
                    out_str += f'No prediction of properties: {target[uuid]["name"]}\n\n'
                    num_errors += 1
                else:
                    t = target[uuid]["properties"]
                    p = prediction[uuid]["properties"]

                    for key in t:
                        if key not in p:
                            out_str += f'Missing key: {key} for {target[uuid]["name"]}\n\n'
                            num_errors += 1
                        else:
                            if type(t[key]) in [list, tuple] and type(p[key]) in [list, tuple]:
                                # We don't care it is a list/tuple
                                if list(t[key]) != list(p[key]):
                                    out_str += f"Difference in {key} of {target[uuid]['name']}:\n"
                                    out_str += f"Prediction: {p[key]}\n"
                                    out_str += f"Target: {t[key]}\n\n"
                                    num_errors += 1
                                else:
                                    out_str += f"correct prediction in {key} of {target[uuid]['name']}: \n"
                                    out_str += f"Prediction: {p[key]}\n"
                                    out_str += f"Target: {t[key]}\n\n"
                                    num_correct += 1
                            elif type(t[key]) == dict and type(p[key]) == dict:
                                if not compare_dict(t[key], p[key]):
                                    out_str += f"Difference in {key} of {target[uuid]['name']}:\n"
                                    out_str += f"Prediction: {p[key]}\n"
                                    out_str += f"Target: {t[key]}\n\n"
                                    num_errors += 1
                                else:
                                    out_str += f"correct prediction in {key} of {target[uuid]['name']}:\n"
                                    out_str += f"Prediction: {p[key]}\n"
                                    out_str += f"Target: {t[key]}\n\n"
                                    num_correct += 1
                            else:
                                if t[key] != p[key]:
                                    out_str += f"Difference in {key} of {target[uuid]['name']}:\n"
                                    out_str += f"Prediction: {p[key]}\n"
                                    out_str += f"Target: {t[key]}\n\n"
                                    num_errors += 1
                                else:
                                    out_str += f"correct prediction in {key} of {target[uuid]['name']}:\n"
                                    out_str += f"Prediction: {p[key]}\n"
                                    out_str += f"Target: {t[key]}\n\n"
                                    num_correct += 1

                if "contains" not in prediction[uuid]:
                    out_str += f'No prediction of contains: {target[uuid]["name"]}\n\n'
                    num_errors += 1
                else:
                    # We don't care the order of contains
                    if sorted(target[uuid]["contains"]) != sorted(prediction[uuid]["contains"]):
                        out_str += f"Difference in contains of {target[uuid]['name']}:\n"
                        out_str += f"Prediction: {prediction[uuid]['contains']}\n"
                        out_str += f"Target: {target[uuid]['contains']}\n\n"
                        num_errors += 1
                    else:
                        out_str += f"correct prediction in contains of {target[uuid]['name']}:\n"
                        out_str += f"Prediction: {prediction[uuid]['contains']}\n"
                        out_str += f"Target: {target[uuid]['contains']}\n\n"
                        num_correct += 1


        # compare scores
        if evaluate_score:
            for key in score_t:
                if key not in score_p:
                    out_str += f'Missing key: {key} for scoring\n\n'
                    num_errors_score += 1
                else:
                    if score_t[key] != score_p[key]:
                        out_str += f"Difference in {key} for scoring:\n"
                        out_str += f"Prediction: {score_p[key]}\n"
                        out_str += f"Target: {score_t[key]}\n\n"
                        num_errors_score += 1
                    else:
                        out_str += f"correct prediction in {key} for scoring:\n"
                        out_str += f"Prediction: {score_p[key]}\n"
                        out_str += f"Target: {score_t[key]}\n\n"
                        num_correct_score += 1

    except:
        out_str = "Wrong prediction format"
        if evaluate_score:
            return -1, -1, -1, -1, out_str
        else:
            return -1, out_str
    if evaluate_score:
        out_str += f"Total errors: {num_errors}, Total score errors: {num_errors_score}\n"
    else:
        out_str += f"Total errors: {num_errors}\n"
    out_str += "-------------------------------------------------------------------\n"

    if evaluate_score:
        return num_errors, num_errors_score, num_correct, num_correct_score, out_str
    else:
        return num_errors, out_str


def compute_score_(solution_str, ground_truth, extra_info, text_game_reward_type="binary"):
    if "The final answer is:" in solution_str:
        answer = solution_str.split("The final answer is:")[-1].strip()
    elif "the final answer is:" in solution_str:
        answer = solution_str.split("the final answer is:")[-1].strip()
    elif "The final state is:" in solution_str:
        answer = solution_str.split("The final state is:")[-1].strip()
    elif "the final state is:" in solution_str:
        answer = solution_str.split("the final state is:")[-1].strip()
    elif "The final game state is:" in solution_str:
        answer = solution_str.split("The final game state is:")[-1].strip()
    elif "the final game state is:" in solution_str:
        answer = solution_str.split("the final game state is:")[-1].strip()
    else:
        pattern = r'```json\s*([\s\S]*?)\s*```'
        matches = re.findall(pattern, solution_str, re.DOTALL)

        if len(matches) == 0 or len(matches[-1]) < 10:
            answer = solution_str
        else:
            answer = matches[-1]

    answer = answer.replace("True", "true")
    answer = answer.replace("False", "false")
    answer = answer.replace("None", "null")
    correct_json_str = repair_json(answer)
    prediction = dirtyjson.loads(correct_json_str)

    if "modified" not in prediction or "removed" not in prediction:
        return 0

    prediction = recover_game_state_from_partial(json.loads(extra_info["data_state"]), prediction, has_score=True)
    num_errors, num_score_errors, num_correct, num_correct_score, eval_out_str = evaluate(prediction,
                                                                                          json.loads(ground_truth),
                                                                                          extra_info["data_action"],
                                                                                          evaluate_score=True)

    if num_errors < 0:  # 格式出错
        return 0
    else:
        item_score = num_correct / (num_correct + num_errors)
        score_score = num_correct_score / (num_correct_score + num_score_errors)

        if text_game_reward_type == "binary":
            return int(item_score == 1 and score_score == 1)

        # only consider gold states
        correct_gold_num_record = []
        curr_state = json.loads(extra_info["data_state"])
        gold_state = json.loads(ground_truth)
        predicted_state = prediction
        diffs_gold = get_state_diff_detail_v2(curr_state, gold_state)

        # process gold state
        gold_stat = {}
        for _, obj2 in diffs_gold["added"]:
            gold_stat[obj2["uuid"]] = {'contains': 1}
            for key in obj2["properties"]:
                gold_stat[obj2["uuid"]][key] = 'na'
        for obj1, _ in diffs_gold["removed"]:
            gold_stat[obj1["uuid"]] = {'contains': 1}
            for key in obj1["properties"]:
                gold_stat[obj1["uuid"]][key] = 'na'
        for key, state_1, state_2, state_code in diffs_gold["modified"]:
            if state_2 is not None:
                uuid = state_2["uuid"]
            else:
                uuid = state_1["uuid"]

            if uuid not in gold_stat:
                gold_stat[uuid] = {}

            gold_stat[uuid][key] = 0 if state_code == 1 else 1

        try:
            diffs = get_state_diff_detail_v2(gold_state, predicted_state)
        except Exception as e:
            wrong_output_format = True
        else:
            # for objects that are added or removed, the per property status are assigned "na", thus are ignored
            for key, state_1, state_2, state_code in diffs["modified"]:
                if state_1 is not None:
                    uuid = state_1['uuid']
                else:
                    uuid = state_2['uuid']

                if gold_stat[uuid][key] == 1:
                    if state_code == 1:
                        correct_gold_num_record.append(1)
                    else:
                        correct_gold_num_record.append(0)

        if len(correct_gold_num_record) == 0:
            return item_score * 0.09 + score_score * 0.01 + 0.2 * int(item_score == 1 and score_score == 1)
        else:
            return item_score * 0.09 + score_score * 0.01 + 0.2 * int(item_score == 1 and score_score == 1) + sum(
                correct_gold_num_record) / len(correct_gold_num_record)


def compute_score(solution_str, ground_truth, extra_info, text_game_reward_type="binary"):
    try:
        return compute_score_(solution_str, ground_truth, extra_info, text_game_reward_type)
    except:
        return 0

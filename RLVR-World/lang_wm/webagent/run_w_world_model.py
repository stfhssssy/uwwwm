
"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse
import collections
import copy
import glob
import heapq
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import openai
import requests
import torch
from PIL import Image
from urllib.parse import urlparse
from agent import (
    PromptAgent,
    construct_agent,
    value_function,

)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent, create_goto_url_action
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router, image_utils

from dotenv import load_dotenv
assert load_dotenv()

DATASET = os.environ["DATASET"]

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )

    parser.add_argument(
        "--total_indices", type=int, required=True, help="Total number of indices"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt", choices=["prompt", "search", "world_model", "baseline"])
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--state_prediction_prompt_path",
        type=str,
        default="search-agents/agent/prompts/jsons/state_prediction/text_only_acctree_format.json"
    )
    parser.add_argument(
        "--value_function_prompt_path",
        type=str,
        default="/agent/prompts/jsons/value_function/text_only_description_format.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--test_config_base_dir", type=str)

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )
    parser.add_argument(
        "--next_state_format",
        type=str,
        default="description_with_tao",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )

    parser.add_argument("--world_model_training", action="store_true", help="Enable world model training")
    parser.add_argument("--world_model_name", type=str, default="none", help="Name of the world model")
    parser.add_argument("--world_model_url", type=str, default="none", help="URL of the world model")

    parser.add_argument("--value_model_training", action="store_true", help="Enable value model training")
    parser.add_argument("--value_model_name", type=str, default="none", help="Name of the value model")
    parser.add_argument("--value_model_url", type=str, default="none", help="URL of the value model")

    # search config
    parser.add_argument("--max_depth", type=int, default=4, help="Max depth for search agents.")
    parser.add_argument("--branching_factor", type=int, default=5, help="Branching factor at each step for the search agent.")
    parser.add_argument("--search_algo", type=str, default="vf", help="Search algorithm to use", choices=["vf", "bfs", "dfs"])
    parser.add_argument("--vf_budget", type=int, default=20, help="Budget for the number of value function evaluations.")
    parser.add_argument("--value_function", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="What value function to use.")

    # example config
    parser.add_argument("--test_idx", type=str, default=None, help="Idx to test")
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)
    parser.add_argument("--my_world_model", action="store_true", help="Use my world model")

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
        ]
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


from evaluation_harness import CaptioningFn


def test(
    args: argparse.Namespace,
    config_file_list: list[str]
) -> None:
    scores = []
    max_steps = args.max_steps
    branching_factor = args.branching_factor
    assert args.vf_budget is not None, "Value function budget should be specified."

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    caption_image_fn: CaptioningFn | None
    eval_caption_image_fn: CaptioningFn | None

    if args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    if DATASET == 'visualwebarena':
        if (
            caption_image_fn
            and args.eval_captioning_model == args.captioning_model
        ):
            eval_caption_image_fn = caption_image_fn
        else:
            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                torch.float16
                if (
                    torch.cuda.is_available()
                    and args.eval_captioning_model_device == "cuda"
                )
                else torch.float32,
                args.eval_captioning_model,
            )
    else:
        caption_image_fn = None
        eval_caption_image_fn = None

    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )  # NOTE: captioning_fn here is used for captioning input images.

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    # check if results file exists
    for config_file in config_file_list:
        # check if html files directory exists and create if not
        if not os.path.exists(os.path.join(args.result_dir, "html_files")):
            os.makedirs(os.path.join(args.result_dir, "html_files"))
        render_helper = RenderHelper(
                config_file, os.path.join(args.result_dir, "html_files"), args.action_set_tag
            )
        # try:
        if True:
            # Load task.
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                image_paths = _c.get("image", None)
                images = []

                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "-m",
                            "browser_env.auto_login",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

                # Load input images for the task, if any.
                if image_paths is not None:
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        # Load image either from the web or from a local path.
                        if image_path.startswith("http"):
                            input_image = Image.open(requests.get(image_path, stream=True).raw)
                        else:
                            input_image = Image.open(image_path)

                        images.append(input_image)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            action_history = []  # Save the action history for the agent so that we can backtrack.
            try:
                obs, info = env.reset(options={"config_file": config_file})
            except:
                update_result(result_file_path, current_idx, 0, error=True)
                return
            state_info: StateInfo = {"observation": obs, "info": info, "url": env.page.url}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            step_idx = 0
            while True:
                step_idx += 1
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    res = agent.next_action( # type: ignore[assignment]
                        trajectory,
                        intent,
                        images=images,
                        meta_data=meta_data,
                        branching_factor=branching_factor
                    )
                    action, action_candidates, next_state_predictions, value_scores, raw_response_for_value_score_calculation = res


                all_candidates = []
                best_actions = [action]
                best_score = None

                stop_trajectory = False

                prev_url = env.page.url
                # Now we can actually execute the best action.
                for best_idx, action in enumerate(best_actions):
                    all_candidates.append(f"Selected action {best_idx}: {action['raw_prediction']}")
                    trajectory.append(action)

                    action_str = get_action_description(
                        action,
                        state_info["info"]["observation_metadata"],
                        action_set_tag=args.action_set_tag,
                        prompt_constructor=agent.prompt_constructor
                        if isinstance(agent, PromptAgent)
                        else None,
                    )
                    render_helper.render(
                        action, state_info, meta_data, args.render_screenshot, all_candidates if args.agent_type == "search" else None, action_candidates, next_state_predictions, value_scores,
                        raw_response_for_value_score_calculation
                    )

                    meta_data["action_history"].append(action_str)

                    if action["action_type"] == ActionTypes.STOP:
                        stop_trajectory = True
                        break

                    obs, _, terminated, _, info = env.step(action)
                    # Save the committed action to the action history.
                    action_history.append(action)
                    curr_url = env.page.url
                    if curr_url != prev_url:
                        # URL has changed, simplify the action_history so that we resume from this checkpoint
                        action_history = [create_goto_url_action(curr_url)]
                        prev_url = curr_url
                    state_info = {"observation": obs, "info": info, "url": env.page.url}
                    trajectory.append(state_info)

                    if terminated:
                        # add an action placeholder
                        print(f"Terminated!")
                        trajectory.append(create_stop_action(""))
                        stop_trajectory = True
                        break

                # We solved the task and can quit.
                if stop_trajectory or (best_score is not None and best_score == 1.0) or early_stop_flag:
                    # Save obs
                    break
            # END SEARCH

            # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
            if DATASET == "visualwebarena":
                assert eval_caption_image_fn is not None
            evaluator = evaluator_router(
                config_file, captioning_fn=eval_caption_image_fn
            )
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page
            )

            scores.append(score)

            test_index = int(config_file.split("/")[-1].replace(".json",""))
            update_result(os.path.join(args.result_dir, "results.json"), test_index, score)
            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")


            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        render_helper.close()

    env.close()
    if len(scores):
        logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_file = os.path.join(result_dir, "results.json")
    with open(result_file, "r") as f:
        result_data = json.load(f)
    task_ids = list(result_data["info"].keys())
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs

def parse_task_id_from_path(config_file_string: str)->int:
    return config_file_string.split("/")[-1].split(".")[0]

def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")

def update_result(result_file_path: str, test_index: int, result: int, error: str = None) -> None:
    with open(result_file_path, "r") as f:
        # deepcopy the result
        import copy
        result_data = copy.deepcopy(json.load(f))
    # add progress
    result_data['info'][test_index] = result
    if not error:
        result_data['progress']['finished'] = len(result_data['info']) - result_data['progress']['error']
    else:
        result_data['progress']['error'] += 1
    result_data['progress']['unfinished'] = result_data['progress']['total'] - len(result_data['info'])
        # Avoid division by zero
    if result_data['progress']['total'] > 0:
        result_data['progress']['percentage'] = (result_data['progress']['finished'] + result_data['progress']['error']) / result_data['progress']['total']
    else:
        result_data['progress']['percentage'] = 0.0

    valid_result_list = [v for v in list(result_data['info'].values()) if v is not None]
    if len(valid_result_list) > 0:
        result_data['overall'] = sum(valid_result_list)/ len(valid_result_list)
    else:
        result_data['overall'] = 0.0

    # sort the result dictionary by test_index in an increasing order

    sorted_info = {str(k): result_data['info'][k] for k in sorted(result_data['info'], key=lambda x: int(x))}

    result_data['info'] = sorted_info
    print(f"################## Updated overall pass ratio: {result_data['overall']}")
    with open(result_file_path, "w") as f:
        json.dump(result_data, f, indent=4)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    test_file_list = []
    result_file_path = os.path.join(args.result_dir, "results.json")
    # check if result file exists
    if not os.path.exists(result_file_path):
        with open(result_file_path, "w") as f:
            json.dump({"overall": None, "progress": {"total": args.total_indices, "unfinished": args.total_indices, "finished": 0, "error": 0, "percentage": 0}, "info": {}}, f, indent=4)
    if args.test_idx is not None:
        print(f"Testing on {args.test_idx}")

        for x in args.test_idx.split(","):
            test_file_list.append(os.path.join(test_config_base_dir, f"{x}.json"))
    else:
        print(f"Testing on {args.test_start_idx} to {args.test_end_idx}")
        st_idx = args.test_start_idx
        ed_idx = args.test_end_idx
        for i in range(st_idx, ed_idx):
            test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    test_file_list = get_unfinished(test_file_list, args.result_dir)
    iter_count = 0
    from colorama import Fore, Style

    while (len(test_file_list) > 0) and (iter_count < 5):
        print(f"{Fore.CYAN}############# Iteration: {iter_count} #############{Style.RESET_ALL}")

        print(f"Total {len(test_file_list)} tasks left")
        args.render = False
        args.render_screenshot = True
        args.save_trace_enabled = True

        args.current_viewport_only = True
        dump_config(args)

        test(args, test_file_list)

        test_file_list = get_unfinished(test_file_list, args.result_dir)

    leftover_config_files = get_unfinished(test_file_list, args.result_dir)
    if len(leftover_config_files) > 0:
        for left_cfg in leftover_config_files:
            current_idx = parse_task_id_from_path(left_cfg)
            update_result(result_file_path, current_idx, 0, error=True)





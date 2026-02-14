# This file is changed from WMA project:
# https://github.com/kyle8581/WMA-Agents

import json
import random
import re
import os
import time

import numpy as np

from typing import Any, Optional
from beartype import beartype

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import StateInfo
from browser_env.helper_functions import get_action_description
from browser_env.env_config import URL_MAPPINGS
from filelock import FileLock
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .agent import Agent, PromptAgent
from agent.prompts.prompt_constructor import CoTPromptConstructor
from transformers import AutoTokenizer  # type: ignore
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage

global_base_url = os.environ["OPENAI_API_BASE"]
def extract_sections(text):
    pattern = re.compile(
        r"New items:\n(.*?)(?=\n\nDeleted items:)|"  # New items
        r"Deleted items:\n(.*?)(?=\n\nUpdated items:)|"  # Deleted items
        r"Updated items:\n(.*?)(?=\n\n|\Z)",  # Updated items
        re.DOTALL
    )

    results = {
        "new_items": None,
        "deleted_items": None,
        "updated_items": None
    }

    for match in pattern.finditer(text):
        # New items
        if match.group(1) is not None:
            content = match.group(1).strip()
            results["new_items"] = content if content != "None" else None

        # Deleted items
        elif match.group(2) is not None:
            content = match.group(2).strip()
            results["deleted_items"] = content or None

        # Updated items
        elif match.group(3) is not None:
            content = match.group(3).strip()
            results["updated_items"] = content or None

    return results

def safe_json_load(path, max_retries=100, min_delay=0.05, max_delay=0.2):
    for attempt in range(max_retries):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"[Attempt {attempt+1}/{max_retries}] JSONDecodeError: {e}")
        except Exception as e:
            print(f"[Attempt {attempt+1}/{max_retries}] Other error: {e}")
        time.sleep(random.uniform(min_delay, max_delay))
    raise RuntimeError(f"Failed to read valid JSON from {path} after {max_retries} attempts.")

class WMAgent(Agent):
    """
    World Model Agent Process:
    1. Sample Multiple Actions:
       Generate a set of potential actions.
    
    2. Predict Next States:
       For each generated action, use the world model to predict the resulting state.
    
    3. Calculate Rewards:
       Evaluate the reward for each action based on its predicted next state.
    
    4. Select Best Action:
       Choose the action with the highest calculated reward.
    """
    @beartype
    def __init__(
        self,
        agent_type: str,
        branching_factor: int,
        action_set_tag: str,
        vf_budget: int,
        model_name: str,
        action_prediction_prompt_path: str,
        state_prediction_prompt_path: str,
        value_function_prompt_path: str,
        world_model_training: bool,
        world_model_name: str = None,
        world_model_url: str = None,
        value_model_training: bool = False,
        value_model_name: str = None,
        value_model_url: str = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.agent_type = agent_type
        self.prompt_constructor = CoTPromptConstructor
        self.model_name = model_name
        self.branching_factor = branching_factor
        self.vf_budget = vf_budget
        self.top_p = top_p
        self.temperature = temperature
        self.raw_response_stack = []
        self.intent_stack = []
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.action_prediction_prompt_path = action_prediction_prompt_path
        self.action_prediction_template = safe_json_load(action_prediction_prompt_path)
        self.state_prediction_prompt_path = state_prediction_prompt_path
        self.state_prediction_template = safe_json_load(state_prediction_prompt_path)
        self.value_function_prompt_path = value_function_prompt_path
        self.world_model_training = world_model_training
        self.world_model_name = world_model_name
        self.world_model_url = world_model_url
        self.value_model_training = value_model_training
        self.value_model_name = value_model_name
        self.value_model_url = value_model_url
        self.action_set_tag = action_set_tag
        self.refine_tao_prompt_path = "agent/prompts/jsons/refine_tao.json"
        self.translate_prompt_path = "agent/prompts/jsons/rlvr_translate_prompt.json"

        if self.agent_type == "world_model":
            self.policy_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                top_p=self.top_p,
                temperature=self.temperature
            )
        else:
            self.policy_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                n=self.branching_factor,
                top_p=self.top_p,
                temperature=self.temperature
            )

        print(self.world_model_training.__class__, self.world_model_training)
        if self.world_model_training:
            self.prediction_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name = self.world_model_name,
                base_url=self.world_model_url,
                top_p=self.top_p,
                temperature=self.temperature,
            )
        else:
            self.prediction_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                top_p=self.top_p,
                temperature=self.temperature,
                base_url = global_base_url,
            )
        
        print(self.value_model_training.__class__, self.value_model_training)
        if self.value_model_training:
            self.value_function_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.value_model_name,
                base_url=self.value_model_url,
                top_p=self.top_p,
                temperature=self.temperature,
                n=self.vf_budget
            )
        else:
            self.value_function_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                top_p=self.top_p,
                temperature=self.temperature,
                base_url=global_base_url,
            )

        self.prompt_function_llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                top_p=self.top_p,
                temperature=self.temperature,
                base_url=global_base_url,
        )

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def get_current_observation(self, trajectory: Trajectory) -> str:
        return trajectory[-1]["observation"]['text']

    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        branching_factor: int = 5,
    ):
        state_info: StateInfo = trajectory[-1]
        obs = state_info["observation"]
        page = state_info["info"]["page"]
        raw_url = page.url
        current_url = self.map_url_to_real(url=raw_url)

        # Step 1: Sample Mulitple actions.
        # Step 2: For each generated action, we predict the next state with the world model.
        # Step 3: We calculate the value for each action based on the predicted next state.
        # Step 4: We select the action with the highest value.

        # ==============================
        # Step 1 : Sample multiple actions.
        # ==============================
        print("#################### GENERATE MULTIPLE ACTIONS ########################\n")
        action_prediction_prompt = self.generate_prompt(self.action_prediction_template)

        if (len(meta_data["action_history"]) != 0):
            previous_action_str = "\n".join(meta_data["action_history"])
        elif (len(meta_data["action_history"]) == 0):
            previous_action_str = "None"

        input_variables_for_action = {
            "objective": intent,
            "url": current_url,
            "observation": obs['text'],
            "previous_action": previous_action_str
        }

        action_generation_input = action_prediction_prompt.invoke(input_variables_for_action)
        responses = []
        for i in range(max(self.branching_factor*2, 20)):
            print(f"Generating action {i+1}...")
            raw_response_for_action_prediction = self.policy_llm.generate([action_generation_input])
            responses.append(raw_response_for_action_prediction.generations[0][0].text)
        
        all_actions = {}
        parsed_actions_count = {}
        id = 0
        for response in responses:
            parsed_response = self.extract_action(response)

            print(f"Parsed action {id}: {parsed_response}")
            id += 1

            if parsed_response in all_actions: # when we get the same action, we increment the count.
                parsed_actions_count[parsed_response] += 1

            else: # when we get a new action, we create a new action instance.
                try:
                    if self.action_set_tag == "id_accessibility_tree":
                        action = create_id_based_action(parsed_response)
                    elif self.action_set_tag == "playwright":
                        action = create_playwright_action(parsed_response)
                    elif self.action_set_tag == "som":
                        action = create_id_based_action(parsed_response)
                    else:
                        raise ValueError(
                            f"Unknown action type {self.action_set_tag}"
                        )
                except Exception:
                    action = create_none_action()

                parsed_actions_count[parsed_response] = 1
                action["raw_prediction"] = response
                all_actions[parsed_response] = action

        top_actions = sorted(parsed_actions_count, key=parsed_actions_count.get, reverse=True)[:branching_factor]
        top_action_count = sum([parsed_actions_count[action] for action in top_actions])
        updated_actions = []
        for action in top_actions:
            a = all_actions[action]
            a['prob'] = parsed_actions_count[action] / top_action_count
            updated_actions.append(a)

        top_action_str = []
        for action_index, action_str in enumerate(top_actions):
            try:
                action_ = create_id_based_action(action_str)
            except:
                action_ = create_none_action()

            action_["raw_prediction"] = responses[action_index]
            action_str = get_action_description(
                    action_,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=self.action_set_tag,
                    prompt_constructor=self.prompt_constructor
                    if isinstance(self, PromptAgent)
                    else None,
            )
            top_action_str.append(action_str)

        for action_index, stop_action in enumerate(top_actions):
            try:
                action_ = create_id_based_action(stop_action)
            except:
                action_ = create_none_action()

            action_["raw_prediction"] = responses[action_index]
            if "stop" in stop_action:
                raw_response_for_state_prediction = "None"
                raw_response_for_value_score_calculation = "None"
                value_scores = 1.0
                return (
                    action_,
                    [stop_action],
                    [raw_response_for_state_prediction],
                    [value_scores],
                    [[raw_response_for_value_score_calculation]]
                )
        
        # ==============================
        # Step 2: For each generated action, we predict the next state with the world model.
        # ==============================
        print("#################### PREDICT NEXT STATE ########################\n")
        state_prediction_prompt = self.generate_prompt(self.state_prediction_template)
        state_prediction_chain = state_prediction_prompt | self.prediction_llm

        if (len(meta_data["action_history"]) != 0):
            previous_action_str = "\n".join(meta_data["action_history"])
        elif (len(meta_data["action_history"]) == 0):
            previous_action_str = "None"
        max_input_tokens = 5000
        multiple_input_for_state = []
        print(f"Top actions: {top_actions}")
        for action_ in top_actions:
            input_variable = {
                "objective": intent,
                "url": current_url,
                "previous_action": previous_action_str,
                "observation": obs['text'],
                "current_action": action_
            }
            # prompt = self.generate_prompt(self.state_prediction_template).format(**input_variable)
            # input_ids = qwen_tokenizer(prompt)["input_ids"]
            # if len(input_ids) <= max_input_tokens:
            multiple_input_for_state.append(input_variable)

        print(f"{len(multiple_input_for_state)} inputs to predict...")
        raw_response_for_state_prediction = state_prediction_chain.batch(multiple_input_for_state)
        parsed_state = []
        for response in raw_response_for_state_prediction:
            next_state = response.content # will be parsed later
            parsed_state.append(next_state)

        # ==============================
        # Step 3: We calculate the reward for each action based on the predicted next state.
        # ==============================
        print("#################### CALCULATE VALUE ########################\n")
        value_scores, raw_response_for_value_score_calculation = self.value_function(top_action_str, parsed_state, previous_action_str, intent, trajectory)

        # ==============================
        # Step 4: We select the action with the highest reward.
        # ==============================
        best_action_index = np.argmax(value_scores)

        try:
            action = create_id_based_action(top_actions[best_action_index])
        except:
            action = create_none_action()
        action["raw_prediction"] = responses[best_action_index]

        return (
            action,
            top_actions,
            [ns.content for ns in raw_response_for_state_prediction],
            value_scores,
            raw_response_for_value_score_calculation
        )

    def value_function(
        self,
        predicted_actions: list[str],
        predicted_next_states: list[str],
        previous_action_str: str,
        objective: str,
        trajectory: Trajectory
    ) -> list[float]:
        print(f"Raw prediction: {predicted_next_states}")
        print("#################### REFINING TAO ########################\n")
        refine_tao_prompt = safe_json_load(self.refine_tao_prompt_path)
        refine_tao_prompt = self.generate_prompt(refine_tao_prompt)

        refined_taos = []

        for prediction in predicted_next_states:
            parsed_state = extract_sections(prediction) # will be the input for refining tao
            refine_tao_input = refine_tao_prompt.format_messages(**parsed_state)
            raw_response_for_refined_tao = self.prompt_function_llm.generate([refine_tao_input])
            refined_taos.append(raw_response_for_refined_tao.generations[0][0].text)
        print(f"Refined taos: {refined_taos}")

        print("#################### TRANSLATE ########################\n")
        translate_prompt = safe_json_load(self.translate_prompt_path)
        translate_prompt = self.generate_prompt(translate_prompt)

        translated_taos = []

        for action_index, ns in enumerate(predicted_next_states):
            url = trajectory[-1]['url']
            input_variables = {
                "url": url,
                "objective": objective,
                "prev_action": previous_action_str,
                "cur_action": predicted_actions[action_index],
                "cur_observation": self.get_current_observation(trajectory),
                "tao": refined_taos[action_index]
            }
            translate_input = translate_prompt.format_messages(**input_variables)
            translated_tao = self.prompt_function_llm.generate([translate_input]).generations[0][0].text
            translated_taos.append(translated_tao)
            print(f"Translated taos: {translated_taos}")\
            # set the translated taos to be the predicted next states
            predicted_next_states = translated_taos

        prompt_template = safe_json_load(self.value_function_prompt_path)

        value_function_prompt = self.generate_prompt(prompt_template)

        all_value_scores = []
        all_raw_responses = []
        for action_index, ns in enumerate(predicted_next_states):
            url = trajectory[-1]['url']
            input_variables = {
                "url": url,
                "objective": objective,
                "previous_action": previous_action_str,
                "current_action": predicted_actions[action_index],
                "observation": self.get_current_observation(trajectory),
                "next_state_prediction": ns
            }
            # deepseek model do not support n > 1, so we have to generate multiple times
            all_responses = []
            for i in range(self.vf_budget):
                value_function_input = value_function_prompt.invoke(input_variables)
                raw_response_for_value_calculation = self.value_function_llm.generate([value_function_input])
                all_responses.append(raw_response_for_value_calculation.generations[0][0].text)

            if self.value_model_training:
                calculated_value_scores, all_individual_value_scores = self.process_mean_value_score_for_value_model(all_responses)
            else:
                calculated_value_scores, all_individual_value_scores = self.process_mean_value_score_likert(all_responses)
            all_value_scores.append(calculated_value_scores)
            all_raw_responses.append(
                [f"Score: {individual_value_score} | {raw_response}" for raw_response, individual_value_score in zip(all_responses, all_individual_value_scores)]
            )

        return all_value_scores, all_raw_responses

    def generate_prompt(self, prompt_template: dict[str, Any]) -> ChatPromptTemplate:
        system_message = prompt_template['intro']
        examples = prompt_template['examples']
        template = prompt_template['template']

        messages = [("system", system_message)]
        if prompt_template != self.state_prediction_template or not self.world_model_training:
            for i in range(len(examples)):
                messages.extend([
                    ("user", examples[i][0]),
                    ("assistant", examples[i][1]),
                ])
        messages.append(("user", template))
        final_prompt = ChatPromptTemplate.from_messages(messages)
        return final_prompt

    def reset(self, test_config_file: str) -> None:
        pass

    def flush_stacks(self):
        self.state_image_stack = []
        self.raw_response_stack = []
        self.intent_stack = []

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def extract_action(self, response: str) -> str:
        action_splitter = self.action_prediction_template["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"

        action_splitter_2 = "`"
        pattern2 = rf"{action_splitter_2}((.|\n)*?){action_splitter_2}"

        match = re.search(pattern, response)
        match_2 = re.search(pattern2, response)
        if match:
            return match.group(1).strip()
        elif match_2:
            return match_2.group(1).strip()
        else:
            print(f"Cannot parse action from response {response}")
            return "None"

    def extract_state(self, response: str) -> str:
        rationale_pattern = r"\[Rationale\](.*?)\[Next State\]"
        next_state_pattern = r"\[Next State\](.*)"

        rationale_match = re.search(rationale_pattern, response, re.DOTALL)
        next_state_match = re.search(next_state_pattern, response, re.DOTALL)

        rationale = rationale_match.group(1).strip() if rationale_match else ""
        next_state = next_state_match.group(1).strip() if next_state_match else ""

        return rationale, next_state

    def process_mean_value_score(
        self,
        all_responses: list[str],
        should_log: bool = True
    ) -> float:
        """
        Description:
        This method calculates the mean value score from multiple natural language outputs.
        Basically, the LLM is asked to evaluate:
        (1) whether the task is successfully completed at the next state, (i.e. the ouptut contains 'success')
        (2) whether the progress is goring to the correct direction. (i.e. the LLM answers yes to the second question)

        Format:
        input:
        - all_responses (list of strings): list of natural language outputs from the LLM.
        - should_log (boolean): whether to print the intermediate outputs.
        output:
        - score (float): mean value score which is processed from the nl outputs from the LLM.
        """
        all_scores = []
        for r_idx, r in enumerate(all_responses):
            if should_log:
                print(f"Output {r_idx}: {r}")
            try:
                pred = re.search(r'Status: "?(.+)"?', r).group(1)
                if 'success' in pred.lower():
                    score = 1.0
                else:
                    # Check if it's on the path to success
                    on_path = re.search(r'On the right track to success: "?(.+)"?', r).group(1)
                    if 'yes' in on_path.lower():
                        score = 0.5
                    else:
                        score = 0.0
            except Exception as e:
                print(f"Error parsing response: {e}")
                score = 0.0

            all_scores.append(score)

        score = np.mean(all_scores).item()
        if should_log:
            print(f"Final score: {score}")
            print('=' * 30)

        return score, all_scores
    
    def process_mean_value_score_likert(
        self,
        all_responses: list[str],
        should_log: bool = True
    ) -> float:
        """
        Description:
        This method calculates the mean value score from multiple natural language outputs.
        The LLM is asked to evaluate:
        (1) the performance of an action, using the Likert Scale (1 to 5),
        (2) whether the action moves the task towards completion (success or failure).
        
        The method extracts the score from the Likert Scale for each response and computes the average.

        Format:
        input:
        - all_responses (list of strings): list of natural language outputs from the LLM.
        - should_log (boolean): whether to print the intermediate outputs.
        output:
        - score (float): mean value score which is processed from the nl outputs from the LLM.
        """
        all_scores = []
        for r_idx, r in enumerate(all_responses):
            if should_log:
                print(f"Output {r_idx}: {r}")
            try:
                # Extract the Likert Scale score from the response

                score = re.search(r'\[Score\]\:\s*(\d+)', r).group(1)
                score = float(score)

                # Ensure the score is within the valid range (1 to 5)
                if 1.0 <= score <= 5.0:
                    all_scores.append(score)
                else:
                    print(f"Warning: Score out of range in response {r_idx}: {score}")
                    all_scores.append(0.0)  # Invalid score

            except Exception as e:
                print(f"Error parsing response {r_idx}: {e}")
                all_scores.append(0.0)  # Default to 0 if there's a parsing issue

        # Calculate the mean of the scores
        mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        if should_log:
            print(f"Final score: {mean_score}")
            print('=' * 30)

        return mean_score, all_scores
    
    def process_mean_value_score_for_value_model(
        self,
        all_responses: list[str],
        should_log: bool = True
    ) -> float:

        all_scores = []
        valid_scores = []
        for r_idx, r in enumerate(all_responses):
            if should_log:
                print(f"Output {r_idx}: {r}")
            try:
                patterns = [
                    r'\[Score\]\s*([\d\.]+)',
                    r'score of\s*([\d\.]+)',
                    r'Score\:\s*([\d\.]+)',
                    r'[Score\]\:\s*([\d\.]+)'
                ]
                
                score = None
                for pattern in patterns:
                    match = re.search(pattern, r)
                    if match:
                        score = float(match.group(1).rstrip('.'))
                        break

                if score is not None and 0.0 <= score <= 1.0:
                    valid_scores.append(score)
                else:
                    print(f"Warning: Score out of range in response {r_idx}: {score}")
                    score = None

            except Exception as e:
                print(f"Error parsing response {r_idx}: {e}")
                score = None

            all_scores.append(score if score is not None else 0.0)

        # Calculate the mean of the valid scores
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        if should_log:
            print(f"Final score: {mean_score}")
            print('=' * 30)

        return mean_score, all_scores